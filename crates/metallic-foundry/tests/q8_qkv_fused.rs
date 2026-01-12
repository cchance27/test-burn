#![cfg(test)]

use std::time::Instant;

use half::f16;
use serial_test::serial;

use metallic_context::{
    Context, MetalError, kernels::{elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv_qkv_fused::MatmulGemvQkvFusedOp, matmul_mlx::MatMulMlxOp}, tensor::{
        Tensor, TensorStorage, TensorType, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor}
    }
};

fn make_fp16_tensor(ctx: &mut Context<metallic_context::tensor::F16>, dims: Vec<usize>, seed: u64) -> Result<Tensor<metallic_context::tensor::F16>, MetalError> {
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<metallic_context::tensor::F16>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data)
}

fn quantize_q8_0_canonical_from_f32(
    ctx: &Context<metallic_context::tensor::F16>,
    k: usize,
    n: usize,
    weights_kn: &[f32],
) -> Result<(QuantizedQ8_0Tensor, Vec<u8>, Vec<u8>), MetalError> {
    use metallic_context::tensor::Q8_0_WEIGHTS_PER_BLOCK;
    assert_eq!(weights_kn.len(), k * n);
    let blocks_per_k = k.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
    let total_blocks = blocks_per_k * n;
    let mut data = vec![0u8; total_blocks * Q8_0_WEIGHTS_PER_BLOCK];
    let mut scales = vec![0u8; total_blocks * Q8_0_SCALE_BYTES_PER_BLOCK];

    for bk in 0..blocks_per_k {
        let base_k = bk * Q8_0_WEIGHTS_PER_BLOCK;
        let blk_len = Q8_0_WEIGHTS_PER_BLOCK.min(k - base_k);
        for col in 0..n {
            let mut max_abs = 0f32;
            for i in 0..blk_len {
                let v = weights_kn[(base_k + i) * n + col];
                max_abs = max_abs.max(v.abs());
            }
            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let d_bits = f16::from_f32(d).to_bits().to_le_bytes();
            let sb = (bk * n + col) * Q8_0_SCALE_BYTES_PER_BLOCK;
            scales[sb..sb + 2].copy_from_slice(&d_bits);
            let base = (bk * n + col) * Q8_0_WEIGHTS_PER_BLOCK;
            for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                let val = if i < blk_len { weights_kn[(base_k + i) * n + col] } else { 0.0 };
                let q = if d > 0.0 { (val / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                data[base + i] = q as u8;
            }
        }
    }
    let tensor = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx)?;
    Ok((tensor, data, scales))
}

fn assert_close(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let rel = diff / x.abs().max(1.0);
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(rel);
        assert!(
            diff <= 0.10 || rel <= 0.05,
            "{} idx={} diff={} rel={} x={} y={}",
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
fn qkv_fused_parity_vs_fp16() -> Result<(), MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let k = 896usize;
    let n = 896usize;
    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0x1111_2222)?;
    let wq = make_fp16_tensor(&mut ctx, vec![k, n], 0x3333_4444)?;
    let wk = make_fp16_tensor(&mut ctx, vec![k, n], 0x5555_6666)?;
    let wv = make_fp16_tensor(&mut ctx, vec![k, n], 0x7777_8888)?;
    let q_bias = make_fp16_tensor(&mut ctx, vec![n], 0x9999_AAAA)?;
    let k_bias = make_fp16_tensor(&mut ctx, vec![n], 0x8888_BBBB)?;
    let v_bias = make_fp16_tensor(&mut ctx, vec![n], 0x7777_CCCC)?;

    // FP16 refs via MLX GEMM baseline
    let yq_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wq), None, None, false, false, 1.0, 0.0), None)?;
    let yq_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yq_fp16, q_bias.clone()), None)?;
    let yk_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wk), None, None, false, false, 1.0, 0.0), None)?;
    let yk_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yk_fp16, k_bias.clone()), None)?;
    let yv_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wv), None, None, false, false, 1.0, 0.0), None)?;
    let yv_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yv_fp16, v_bias.clone()), None)?;

    let fq = yq_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fk = yk_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fv = yv_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    // Quantize and fused
    let (wq_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wq.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wk_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wk.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wv_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wv.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;

    let y_packed = ctx.call::<MatmulGemvQkvFusedOp>(
        (
            &x,
            (
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wq_q),
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wk_q),
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wv_q),
            ),
            (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
        ),
        None,
    )?;
    ctx.synchronize();
    let elem = metallic_context::tensor::Dtype::F16.size_bytes();
    let yq_tensor = y_packed.build_view(vec![1, n], vec![n, 1], y_packed.offset);
    let yk_tensor = y_packed.build_view(vec![1, n], vec![n, 1], y_packed.offset + n * elem);
    let yv_tensor = y_packed.build_view(vec![1, n], vec![n, 1], y_packed.offset + 2 * n * elem);
    let yq = yq_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let yk = yk_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let yv = yv_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    assert_close(&fq, &yq, "qkv_fused_yq");
    assert_close(&fk, &yk, "qkv_fused_yk");
    assert_close(&fv, &yv, "qkv_fused_yv");
    Ok(())
}

#[test]
#[ignore]
#[serial]
fn bench_qkv_fused_vs_fp16() -> Result<(), MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let k = 896usize;
    let n = 896usize;
    let iters = 1000usize;
    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0xABCD_EF01)?;
    let wq = make_fp16_tensor(&mut ctx, vec![k, n], 0x1111_AAAA)?;
    let wk = make_fp16_tensor(&mut ctx, vec![k, n], 0x2222_BBBB)?;
    let wv = make_fp16_tensor(&mut ctx, vec![k, n], 0x3333_CCCC)?;
    let q_bias = make_fp16_tensor(&mut ctx, vec![n], 0x4444_1111)?;
    let k_bias = make_fp16_tensor(&mut ctx, vec![n], 0x5555_2222)?;
    let v_bias = make_fp16_tensor(&mut ctx, vec![n], 0x6666_3333)?;

    // FP16 baseline: three matmuls per iteration
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wq), Some(&q_bias), None, false, false, 1.0, 0.0), None)?;
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wk), Some(&k_bias), None, false, false, 1.0, 0.0), None)?;
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wv), Some(&v_bias), None, false, false, 1.0, 0.0), None)?;
    }
    ctx.synchronize();
    let t_fp16 = t0.elapsed().as_secs_f64();

    // Q8 fused
    let (wq_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wq.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wk_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wk.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wv_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n, &wv.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulGemvQkvFusedOp>(
            (
                &x,
                (
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wq_q),
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wk_q),
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wv_q),
                ),
                (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
            ),
            None,
        )?;
    }
    ctx.synchronize();
    let t_q8 = t1.elapsed().as_secs_f64();

    println!("QKV FP16 x3: total={:.3}s avg={:.6}ms", t_fp16, 1e3 * t_fp16 / iters as f64);
    println!("QKV Q8 fused: total={:.3}s avg={:.6}ms", t_q8, 1e3 * t_q8 / iters as f64);
    Ok(())
}

#[test]
#[ignore]
#[serial]
fn qkv_fused_parity_vs_fp16_mixed_dims() -> Result<(), MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let k = 896usize;
    let n_q = 896usize;
    let n_kv = 128usize;

    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0x1111_9999)?;
    let wq = make_fp16_tensor(&mut ctx, vec![k, n_q], 0xAAAA_BBBB)?;
    let wk = make_fp16_tensor(&mut ctx, vec![k, n_kv], 0xCCCC_DDDD)?;
    let wv = make_fp16_tensor(&mut ctx, vec![k, n_kv], 0xEEEE_FFFF)?;
    let q_bias = make_fp16_tensor(&mut ctx, vec![n_q], 0x1357_2468)?;
    let k_bias = make_fp16_tensor(&mut ctx, vec![n_kv], 0x2468_1357)?;
    let v_bias = make_fp16_tensor(&mut ctx, vec![n_kv], 0x9999_5555)?;

    // FP16 refs via MLX GEMM baseline
    let yq_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wq), None, None, false, false, 1.0, 0.0), None)?;
    let yq_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yq_fp16, q_bias.clone()), None)?;
    let yk_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wk), None, None, false, false, 1.0, 0.0), None)?;
    let yk_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yk_fp16, k_bias.clone()), None)?;
    let yv_fp16 = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wv), None, None, false, false, 1.0, 0.0), None)?;
    let yv_fp16 = ctx.call::<BroadcastElemwiseAddInplaceOp>((yv_fp16, v_bias.clone()), None)?;
    let fq = yq_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fk = yk_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let fv = yv_fp16.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    // Quantize and fused
    let (wq_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_q, &wq.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wk_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_kv, &wk.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wv_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_kv, &wv.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;

    let y_packed = ctx.call::<MatmulGemvQkvFusedOp>(
        (
            &x,
            (
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wq_q),
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wk_q),
                &metallic_context::tensor::QuantizedTensor::Q8_0(&wv_q),
            ),
            (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
        ),
        None,
    )?;
    ctx.synchronize();
    let elem = metallic_context::tensor::Dtype::F16.size_bytes();
    let yq_tensor = y_packed.build_view(vec![1, n_q], vec![n_q, 1], y_packed.offset);
    let yk_tensor = y_packed.build_view(vec![1, n_kv], vec![n_kv, 1], y_packed.offset + n_q * elem);
    let yv_tensor = y_packed.build_view(vec![1, n_kv], vec![n_kv, 1], y_packed.offset + (n_q + n_kv) * elem);
    let yq = yq_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let yk = yk_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let yv = yv_tensor.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>();

    assert_close(&fq, &yq, "qkv_fused_yq_mixed");
    assert_close(&fk, &yk, "qkv_fused_yk_mixed");
    assert_close(&fv, &yv, "qkv_fused_yv_mixed");
    Ok(())
}

#[test]
#[ignore]
#[serial]
fn bench_qkv_fused_vs_fp16_mixed_dims() -> Result<(), MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let k = 896usize;
    let n_q = 896usize;
    let n_kv = 128usize;
    let iters = 1000usize;

    let x = make_fp16_tensor(&mut ctx, vec![1, k], 0xDEAD_BEEF)?;
    let wq = make_fp16_tensor(&mut ctx, vec![k, n_q], 0x1010_2020)?;
    let wk = make_fp16_tensor(&mut ctx, vec![k, n_kv], 0x3030_4040)?;
    let wv = make_fp16_tensor(&mut ctx, vec![k, n_kv], 0x5050_6060)?;
    let q_bias = make_fp16_tensor(&mut ctx, vec![n_q], 0x0F0F_1111)?;
    let k_bias = make_fp16_tensor(&mut ctx, vec![n_kv], 0x0F0F_2222)?;
    let v_bias = make_fp16_tensor(&mut ctx, vec![n_kv], 0x0F0F_3333)?;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wq), Some(&q_bias), None, false, false, 1.0, 0.0), None)?;
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wk), Some(&k_bias), None, false, false, 1.0, 0.0), None)?;
        let _ = ctx.call::<MatMulMlxOp>((&x, TensorType::Dense(&wv), Some(&v_bias), None, false, false, 1.0, 0.0), None)?;
    }
    ctx.synchronize();
    let t_fp16 = t0.elapsed().as_secs_f64();

    let (wq_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_q, &wq.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wk_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_kv, &wk.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;
    let (wv_q, _, _) = quantize_q8_0_canonical_from_f32(&ctx, k, n_kv, &wv.as_slice().iter().map(|v| v.to_f32()).collect::<Vec<_>>())?;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulGemvQkvFusedOp>(
            (
                &x,
                (
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wq_q),
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wk_q),
                    &metallic_context::tensor::QuantizedTensor::Q8_0(&wv_q),
                ),
                (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
            ),
            None,
        )?;
    }
    ctx.synchronize();
    let t_q8 = t1.elapsed().as_secs_f64();

    println!("QKV FP16 x3: total={:.3}s avg={:.6}ms", t_fp16, 1e3 * t_fp16 / iters as f64);
    println!("QKV Q8 fused: total={:.3}s avg={:.6}ms", t_q8, 1e3 * t_q8 / iters as f64);
    Ok(())
}
