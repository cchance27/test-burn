#![cfg(test)]

use half::f16;
use serial_test::serial;

use crate::{
    Context, MetalError, kernels::{matmul_gemv::MatmulGemvOp, matmul_mlx::MatMulMlxOp}, tensor::{Q8_0_WEIGHTS_PER_BLOCK, Tensor, TensorStorage, TensorType, quantized::QuantizedQ8_0Tensor}
};

#[derive(Clone, Copy)]
struct Shape {
    k: usize,
    n: usize,
    transpose_b: bool,
    name: &'static str,
}

const SHAPES: &[Shape] = &[
    Shape {
        k: 896,
        n: 896,
        transpose_b: true,
        name: "m1_n896_k896",
    },
    Shape {
        k: 896,
        n: 9728,
        transpose_b: true,
        name: "m1_n9728_k896",
    },
    Shape {
        k: 4864,
        n: 896,
        transpose_b: true,
        name: "m1_n896_k4864",
    },
    Shape {
        k: 896,
        n: 1152,
        transpose_b: false,
        name: "m1_n1152_k896",
    },
    Shape {
        k: 896,
        n: 151_936,
        transpose_b: true,
        name: "m1_n151936_k896",
    },
];

fn make_fp16_tensor(ctx: &mut Context<crate::tensor::F16>, dims: Vec<usize>, seed: u64) -> Result<Tensor<crate::tensor::F16>, MetalError> {
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<crate::tensor::F16>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data)
}

fn quantize_q8_0_canonical_from_f32(
    ctx: &Context<crate::tensor::F16>,
    k: usize,
    n: usize,
    weights_kn: &[f32],
) -> Result<QuantizedQ8_0Tensor, MetalError> {
    assert_eq!(weights_kn.len(), k * n);
    let blocks_per_k = (k + Q8_0_WEIGHTS_PER_BLOCK - 1) / Q8_0_WEIGHTS_PER_BLOCK;
    let total_blocks = blocks_per_k * n;
    let mut data = Vec::with_capacity(total_blocks * Q8_0_WEIGHTS_PER_BLOCK);
    let mut scales = Vec::with_capacity(total_blocks * 2);

    for bk in 0..blocks_per_k {
        let base_k = bk * Q8_0_WEIGHTS_PER_BLOCK;
        let blk_len = Q8_0_WEIGHTS_PER_BLOCK.min(k - base_k);
        for col in 0..n {
            // Compute scale = max|x|/127 over the 32‑value block along K for this column
            let mut max_abs = 0f32;
            for i in 0..blk_len {
                let v = weights_kn[(base_k + i) * n + col];
                max_abs = max_abs.max(v.abs());
            }
            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let d_bits = f16::from_f32(d).to_bits().to_le_bytes();
            scales.extend_from_slice(&d_bits);

            for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                let val = if i < blk_len { weights_kn[(base_k + i) * n + col] } else { 0.0 };
                let q = if d > 0.0 { (val / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                data.push(q as u8);
            }
        }
    }

    QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx)
}

fn dense_nt_to_kn(tensor_nt: &Tensor<crate::tensor::F16>, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(tensor_nt.len(), n * k);
    let slice = tensor_nt.as_slice();
    let mut reshaped = vec![0f32; k * n];
    for row in 0..n {
        for col in 0..k {
            let v = slice[row * k + col].to_f32();
            reshaped[col * n + row] = v;
        }
    }
    reshaped
}

fn assert_close_slice(a: &Tensor<crate::tensor::F16>, b: &Tensor<crate::tensor::F16>, name: &str) {
    let a_f: Vec<f32> = a.as_slice().iter().map(|v| v.to_f32()).collect();
    let b_f: Vec<f32> = b.as_slice().iter().map(|v| v.to_f32()).collect();
    assert_eq!(a_f.len(), b_f.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (&x, &y)) in a_f.iter().zip(b_f.iter()).enumerate() {
        let diff = (x - y).abs();
        let rel = diff / (x.abs().max(1.0));
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(rel);
        assert!(
            diff <= 0.10 || rel <= 0.05,
            "{}: idx={} diff={} rel={} x={} y={}",
            name,
            i,
            diff,
            rel,
            x,
            y
        );
    }
    eprintln!("[{}] max_abs={} max_rel={}", name, max_abs, max_rel);
}

#[test]
#[serial]
#[ignore]
fn q8_gemm_parity_non_transposed() -> Result<(), MetalError> {
    // Focus on the non-transposed shape where MLX quant GEMM supports RHS
    let s = Shape {
        k: 896,
        n: 1152,
        transpose_b: false,
        name: "m1_n1152_k896",
    };
    let mut ctx = Context::<crate::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![1, s.k], 0x7777_ABCD)?;
    let b_dense = make_fp16_tensor(&mut ctx, vec![s.k, s.n], 0x1357_2468)?;
    let y_fp16 = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b_dense), None, None, false, false, 1.0, 0.0))?;
    let b_f32: Vec<f32> = b_dense.as_slice().iter().map(|v| v.to_f32()).collect();
    let q8 = quantize_q8_0_canonical_from_f32(&ctx, s.k, s.n, &b_f32)?;
    let y_q8_gemm = ctx.call::<MatMulMlxOp>((
        &a,
        TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8)),
        None,
        None,
        false,
        false,
        1.0,
        0.0,
    ))?;
    ctx.synchronize();
    assert_close_slice(&y_fp16, &y_q8_gemm, s.name);
    Ok(())
}

#[test]
#[serial]
#[ignore]
fn q8_gemv_parity_qwen25_shapes() -> Result<(), MetalError> {
    for s in SHAPES {
        let mut ctx = Context::<crate::tensor::F16>::new()?;
        let a = make_fp16_tensor(&mut ctx, vec![1, s.k], 0x1111_2222)?;
        // Build dense B as (k, n)
        let b_dense = make_fp16_tensor(&mut ctx, vec![s.k, s.n], 0x3333_4444)?;
        // Baseline FP16 (tA=false, tB=false)
        let y_fp16 = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b_dense), None, None, false, false, 1.0, 0.0))?;
        // Quantize B to Q8 canonical
        let b_f32: Vec<f32> = b_dense.as_slice().iter().map(|v| v.to_f32()).collect();
        let q8 = quantize_q8_0_canonical_from_f32(&ctx, s.k, s.n, &b_f32)?;
        // Q8 GEMV result; enable debug for column 768
        let diag_col = 768usize.min(s.n - 1);
        unsafe {
            std::env::set_var("METALLIC_GEMV_DEBUG_COL", diag_col.to_string());
        }
        let y_q8 = ctx.call::<MatmulGemvOp>((&a, TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8)), None))?;
        ctx.synchronize();
        // CPU reference dequant + GEMV (per-block contributions for diag_col)
        let a_f: Vec<f32> = a.as_slice().iter().map(|v| v.to_f32()).collect();
        let dbuf = q8.data.as_slice();
        let sbuf = q8.scales.as_slice();
        let blocks_per_k = (s.k + Q8_0_WEIGHTS_PER_BLOCK - 1) / Q8_0_WEIGHTS_PER_BLOCK;
        let mut y_ref = vec![0f32; s.n];
        let mut y_ref_blocks = vec![0f32; blocks_per_k];
        let mut y_ref_blocks_alt = vec![0f32; blocks_per_k];
        for blk in 0..blocks_per_k {
            let base_k = blk * Q8_0_WEIGHTS_PER_BLOCK;
            for col in 0..s.n {
                let scale_bits = u16::from_le_bytes([sbuf[(blk * s.n + col) * 2], sbuf[(blk * s.n + col) * 2 + 1]]);
                let scale = f16::from_bits(scale_bits).to_f32();
                let base = (blk * s.n + col) * Q8_0_WEIGHTS_PER_BLOCK;
                let mut acc = 0.0f32;
                for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                    let k_idx = base_k + i;
                    if k_idx >= s.k {
                        break;
                    }
                    let q = dbuf[base + i] as i8 as f32;
                    acc = f32::mul_add(q * scale, a_f[k_idx], acc);
                }
                y_ref[col] += acc;
                if col == diag_col {
                    y_ref_blocks[blk] = acc;
                }
            }
        }
        // Alternate column-major interpretation for diagnostics
        for blk in 0..blocks_per_k {
            let base_k = blk * Q8_0_WEIGHTS_PER_BLOCK;
            for col in 0..s.n {
                let scale_idx = (col * blocks_per_k + blk) * 2;
                let scale_bits = u16::from_le_bytes([sbuf[scale_idx], sbuf[scale_idx + 1]]);
                let scale = f16::from_bits(scale_bits).to_f32();
                let base = (col * blocks_per_k + blk) * Q8_0_WEIGHTS_PER_BLOCK;
                let mut acc = 0.0f32;
                for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                    let k_idx = base_k + i;
                    if k_idx >= s.k {
                        break;
                    }
                    let q = dbuf[base + i] as i8 as f32;
                    acc = f32::mul_add(q * scale, a_f[k_idx], acc);
                }
                if col == diag_col {
                    y_ref_blocks_alt[blk] = acc;
                }
            }
        }
        // y_q8 now contains per-block contributions at indices [0..blocks_per_k) for the diag column
        let y_gpu: Vec<f32> = y_q8.as_slice().iter().map(|v| v.to_f32()).collect();
        let gpu_sum: f32 = y_gpu.iter().take(blocks_per_k).sum();
        eprintln!(
            "[PARITY DIAG {}] col={} cpu_ref={} gpu={} diff={}",
            s.name,
            diag_col,
            y_ref[diag_col],
            gpu_sum,
            (y_ref[diag_col] - gpu_sum).abs()
        );
        let mut mismatch = false;
        for blk in 0..blocks_per_k {
            let diff = (y_ref_blocks[blk] - y_gpu[blk]).abs();
            if diff > 1e-3 {
                mismatch = true;
                eprintln!(
                    "[PARITY DIAG {}] block={} cpu_ref={} gpu={} diff={} (base_k={})",
                    s.name,
                    blk,
                    y_ref_blocks[blk],
                    y_gpu[blk],
                    diff,
                    blk * Q8_0_WEIGHTS_PER_BLOCK
                );
            }
        }
        if mismatch {
            eprintln!("[PARITY DIAG {}] gpu_blocks={:?}", s.name, &y_gpu[..blocks_per_k]);
            eprintln!("[PARITY DIAG {}] alt_blocks={:?}", s.name, &y_ref_blocks_alt);
            panic!("kernel parity mismatch");
        }
        // After confirming kernel parity, compare final outputs (optional)
        // reset debug env
        unsafe {
            std::env::remove_var("METALLIC_GEMV_DEBUG_COL");
        }
        // Recompute y_q8 normal (no debug)
        let y_q8_norm = ctx.call::<MatmulGemvOp>((&a, TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8)), None))?;
        ctx.synchronize();
        // Log the diag column for normal run vs CPU quant and FP16
        let y_q8_norm_f: Vec<f32> = y_q8_norm.as_slice().iter().map(|v| v.to_f32()).collect();
        eprintln!(
            "[PARITY DIAG {}] NORMAL col={} cpu_q={} gpu_q={} fp16={} diff_q_fp16={}",
            s.name,
            diag_col,
            y_ref[diag_col],
            y_q8_norm_f[diag_col],
            y_fp16.as_slice()[diag_col].to_f32(),
            (y_q8_norm_f[diag_col] - y_fp16.as_slice()[diag_col].to_f32()).abs()
        );
        assert_close_slice(&y_fp16, &y_q8_norm, s.name);
    }
    Ok(())
}

#[test]
#[serial]
#[ignore]
fn q8_gemm_nt_parity_qwen25_shapes() -> Result<(), MetalError> {
    for s in SHAPES.iter().filter(|shape| shape.transpose_b) {
        let mut ctx = Context::<crate::tensor::F16>::new()?;
        let a = make_fp16_tensor(&mut ctx, vec![1, s.k], 0x6666_7777)?;
        let b_dense = make_fp16_tensor(&mut ctx, vec![s.n, s.k], 0x9999_AAAA)?;
        let y_fp16 = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b_dense), None, None, false, true, 1.0, 0.0))?;
        let b_kn = dense_nt_to_kn(&b_dense, s.n, s.k);
        let q8 = quantize_q8_0_canonical_from_f32(&ctx, s.k, s.n, &b_kn)?;
        let quant = TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8));
        let y_q8 = ctx.matmul(&a, &quant, false, true, None)?;
        ctx.synchronize();
        assert_close_slice(&y_fp16, &y_q8, &format!("{}_nt", s.name));
    }
    Ok(())
}

#[test]
#[serial]
#[ignore]
fn q8_canonical_parity_m2_to_m4_nt() -> Result<(), MetalError> {
    // Validate canonical Q8 path parity vs FP16 for m in {2,3,4} with transpose_b=true
    let ms = [2usize, 3, 4];
    let ns = [512usize, 896, 1536, 2048, 4096, 9728];
    let k = 896usize;

    for &m in &ms {
        for &n in &ns {
            let mut ctx = Context::<crate::tensor::F16>::new()?;
            // A: [m, k]
            let a = make_fp16_tensor(&mut ctx, vec![m, k], 0xCAFE_BABE ^ ((m as u64) << 8) ^ (n as u64))?;
            // B dense NT storage: [n, k], so tB=true yields shape (k,n)
            let b_dense = make_fp16_tensor(&mut ctx, vec![n, k], 0xDEAD_BEEF ^ ((n as u64) << 8) ^ (m as u64))?;
            let y_fp16 = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b_dense), None, None, false, true, 1.0, 0.0))?;
            // Quantize B to canonical (K,N)
            let b_kn = dense_nt_to_kn(&b_dense, n, k);
            let q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &b_kn)?;
            let quant = TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8));
            unsafe {
                std::env::set_var("METALLIC_Q8_CANONICAL_N", "1");
                std::env::set_var("METALLIC_Q8_SMALLM_USE_GEMV", "0");
            }
            let y_q8 = ctx.matmul(&a, &quant, false, true, None)?;
            ctx.synchronize();
            unsafe {
                std::env::remove_var("METALLIC_Q8_CANONICAL_N");
                std::env::remove_var("METALLIC_Q8_SMALLM_USE_GEMV");
            }
            assert_close_slice(&y_fp16, &y_q8, &format!("canonical_m{}_n{}_nt", m, n));
        }
    }
    Ok(())
}

// Removed deprecated row-wise small-m parity test (slow path no longer used)

#[test]
#[serial]
#[ignore]
fn q8_gemm_nt_kernel_parity_m2_to_m4_nt() -> Result<(), MetalError> {
    // Validate dedicated Q8 NT kernel parity vs FP16 for m in {2,3,4}
    let ms = [2usize, 3, 4];
    let ns = [512usize, 896, 1536, 2048, 4096, 9728];
    let k = 896usize;

    for &m in &ms {
        for &n in &ns {
            let mut ctx = Context::<crate::tensor::F16>::new()?;
            let a = make_fp16_tensor(&mut ctx, vec![m, k], 0x2222_4444 ^ ((m as u64) << 8) ^ (n as u64))?;
            let b_dense = make_fp16_tensor(&mut ctx, vec![n, k], 0x6666_8888 ^ ((n as u64) << 8) ^ (m as u64))?;
            let y_fp16 = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b_dense), None, None, false, true, 1.0, 0.0))?;
            let b_kn = dense_nt_to_kn(&b_dense, n, k);
            let q8 = quantize_q8_0_canonical_from_f32(&ctx, k, n, &b_kn)?;
            let quant = TensorType::Quant(crate::tensor::QuantizedTensor::Q8_0(&q8));
            unsafe {
                std::env::set_var("METALLIC_Q8_CANONICAL_N", "0");
                std::env::set_var("METALLIC_Q8_SMALLM_USE_GEMV", "0");
            }
            let y_q8 = ctx.matmul(&a, &quant, false, true, None)?;
            ctx.synchronize();
            unsafe {
                std::env::remove_var("METALLIC_Q8_CANONICAL_N");
                std::env::remove_var("METALLIC_Q8_SMALLM_USE_GEMV");
            }
            assert_close_slice(&y_fp16, &y_q8, &format!("q8_nt_kernel_m{}_n{}_nt", m, n));
        }
    }
    Ok(())
}
