#![cfg(test)]

use std::time::Instant;

use metallic_context::{
    Context, MetalError, kernels::{
        matmul_gemv::{MatmulGemvOp, MatmulGemvSmallMOp, MatmulQ8CanonicalOp, MatmulQ8CanonicalRows16Op, MatmulQ8NtOp}, matmul_mlx::MatMulMlxOp
    }, tensor::{
        Q8_0_WEIGHTS_PER_BLOCK, Tensor, TensorStorage, TensorType, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor}
    }
};
use serial_test::serial;

#[derive(Clone, Copy)]
struct MatmulShape {
    m: usize,
    n: usize,
    k: usize,
    t_a: bool,
    t_b: bool,
    name: &'static str,
}

// Qwen25 common shapes (from tools/metal_probes/MATMUL_QWEN25_SIZES.md)
const SHAPES: &[MatmulShape] = &[
    MatmulShape {
        m: 1,
        n: 896,
        k: 896,
        t_a: false,
        t_b: true,
        name: "m1_n896_k896_tb1",
    },
    MatmulShape {
        m: 1,
        n: 9728,
        k: 896,
        t_a: false,
        t_b: true,
        name: "m1_n9728_k896_tb1",
    },
    MatmulShape {
        m: 1,
        n: 896,
        k: 4864,
        t_a: false,
        t_b: true,
        name: "m1_n896_k4864_tb1",
    },
    MatmulShape {
        m: 1,
        n: 1152,
        k: 896,
        t_a: false,
        t_b: false,
        name: "m1_n1152_k896_tb0",
    },
    MatmulShape {
        m: 1,
        n: 151_936,
        k: 896,
        t_a: false,
        t_b: true,
        name: "m1_n151936_k896_tb1",
    },
];

fn make_fp16_tensor(
    ctx: &mut Context<metallic_context::tensor::F16>,
    dims: Vec<usize>,
    seed: u64,
) -> Result<Tensor<metallic_context::tensor::F16>, MetalError> {
    // Simple deterministic pattern to avoid host RNG overhead
    let len = dims.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        // LCG
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((x >> 32) as u32) as f32 / (u32::MAX as f32) - 0.5;
        data.push(v);
    }
    Tensor::<metallic_context::tensor::F16>::from_f32_slice(dims, TensorStorage::Pooled(ctx), &data)
}

fn make_q8_canonical(
    ctx: &Context<metallic_context::tensor::F16>,
    k: usize,
    n: usize,
    seed: u64,
) -> Result<QuantizedQ8_0Tensor, MetalError> {
    // Canonical layout expects blocks ordered by block_k then n.
    let blocks_per_k = k.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
    let total_blocks = blocks_per_k * n;
    let mut data = Vec::with_capacity(total_blocks * Q8_0_WEIGHTS_PER_BLOCK);
    let mut scales = Vec::with_capacity(total_blocks * Q8_0_SCALE_BYTES_PER_BLOCK);
    let mut x = seed;
    for bk in 0..blocks_per_k {
        for _col in 0..n {
            // scale = 1.0 (fp16 0x3C00)
            scales.extend_from_slice(&0x3C00u16.to_le_bytes());
            for i in 0..Q8_0_WEIGHTS_PER_BLOCK {
                let gi = bk * Q8_0_WEIGHTS_PER_BLOCK + i;
                let q = if gi < k {
                    // deterministic i8 in [-16, 15]
                    x = x.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                    ((x >> 32) as i8) & 0x1f
                } else {
                    0
                };
                data.push(q as u8);
            }
        }
    }
    QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![k, n], &data, &scales, ctx)
}

fn bench_mlx_gemm_fp16(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0xA5A5_1234)?;
    // When transpose_b is true, B should be stored as (n, k) so that the effective shape is (k, n)
    let b_dims = if shape.t_b {
        vec![shape.n, shape.k]
    } else {
        vec![shape.k, shape.n]
    };
    let b = make_fp16_tensor(&mut ctx, b_dims, 0xDEAD_BEEF)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatMulMlxOp>((&a, TensorType::Dense(&b), None, None, shape.t_a, shape.t_b, 1.0, 0.0), None)?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_mlx_gemm_q8(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0xA5A5_1234)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xFEED_BA65)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatMulMlxOp>(
            (
                &a,
                TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(&q8)),
                None,
                None,
                shape.t_a,
                shape.t_b,
                1.0,
                0.0,
            ),
            None,
        )?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_quant_dispatch(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0xABCD_7777)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xFEED_BA5E)?;
    let quant = TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(&q8));
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.matmul(&a, &quant, shape.t_a, shape.t_b, None, None, None)?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_q8_direct_gemv(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0xC0DE_CAFE)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xFACE_FEED)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulGemvOp>(
            (
                &a,
                TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(&q8)),
                false,
                None,
            ),
            None,
        )?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_q8_smallm_kernel_direct(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    // Directly invoke the small-m kernel (rows GEMV). For m==1, use GEMV.
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0x5151_AAAA)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0x6262_BBBB)?;
    let start = Instant::now();
    for _ in 0..iters {
        if shape.m == 1 {
            let _ = ctx.call::<MatmulGemvOp>(
                (
                    &a,
                    TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(&q8)),
                    false,
                    None,
                ),
                None,
            )?;
        } else {
            let _ = ctx.call::<MatmulGemvSmallMOp>((&a, &q8, None), None)?;
        }
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_q8_canonical_direct(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    // Directly invoke the canonical large-N kernel.
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0x9595_EEEE)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xA6A6_FFFF)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulQ8CanonicalOp>((&a, &q8, None), None)?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_q8_canonical_rows16_direct(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    // Directly invoke the 16-row canonical kernel.
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0x9696_7777)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xB6B6_3333)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulQ8CanonicalRows16Op>((&a, &q8, None), None)?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_q8_nt_direct(shape: MatmulShape, iters: usize) -> Result<f64, MetalError> {
    // Directly invoke the Q8 GEMM-NT kernel (supports up to 4 rows).
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let a = make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], 0xB7B7_0001)?;
    let q8 = make_q8_canonical(&ctx, shape.k, shape.n, 0xC8C8_0002)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ctx.call::<MatmulQ8NtOp>((&a, &q8, None), None)?;
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_decode_sequence_fp16(iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let mut inputs = Vec::with_capacity(SHAPES.len());
    let mut weights = Vec::with_capacity(SHAPES.len());
    for (idx, shape) in SHAPES.iter().enumerate() {
        let seed = 0x1111_0000u64.wrapping_add(idx as u64);
        inputs.push(make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], seed)?);
        let b_dims = if shape.t_b {
            vec![shape.n, shape.k]
        } else {
            vec![shape.k, shape.n]
        };
        weights.push(make_fp16_tensor(&mut ctx, b_dims, 0x2222_0000u64.wrapping_add(idx as u64))?);
    }
    let start = Instant::now();
    for _ in 0..iters {
        for (idx, shape) in SHAPES.iter().enumerate() {
            let a = &inputs[idx];
            let b = &weights[idx];
            let dense = TensorType::Dense(b);
            let _ = ctx.matmul(a, &dense, shape.t_a, shape.t_b, None, None, None)?;
        }
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

fn bench_decode_sequence_q8(iters: usize) -> Result<f64, MetalError> {
    let mut ctx = Context::<metallic_context::tensor::F16>::new()?;
    let mut inputs = Vec::with_capacity(SHAPES.len());
    let mut weights = Vec::with_capacity(SHAPES.len());
    for (idx, shape) in SHAPES.iter().enumerate() {
        let seed = 0x3333_0000u64.wrapping_add(idx as u64);
        inputs.push(make_fp16_tensor(&mut ctx, vec![shape.m, shape.k], seed)?);
        weights.push(make_q8_canonical(&ctx, shape.k, shape.n, 0x4444_0000u64.wrapping_add(idx as u64))?);
    }
    let start = Instant::now();
    for _ in 0..iters {
        for (idx, shape) in SHAPES.iter().enumerate() {
            let a = &inputs[idx];
            let q8 = &weights[idx];
            let quant = TensorType::Quant(metallic_context::tensor::QuantizedTensor::Q8_0(q8));
            let _ = ctx.matmul(a, &quant, shape.t_a, shape.t_b, None, None, None)?;
        }
    }
    ctx.synchronize();
    Ok(start.elapsed().as_secs_f64())
}

#[test]
#[serial]
#[ignore]
fn bench_qwen25_shapes_fp16_vs_q8() -> Result<(), MetalError> {
    // NOTE: very large N shapes can be slow for 1000 iterations. For N > 20k, reduce to 200 iterations.
    for &shape in SHAPES {
        let iters = if shape.n > 20_000 { 200 } else { 1000 };
        println!(
            "\n== Shape {}: m={} n={} k={} tA={} tB={} (iters={})",
            shape.name, shape.m, shape.n, shape.k, shape.t_a as u8, shape.t_b as u8, iters
        );

        let t_fp16 = bench_mlx_gemm_fp16(shape, iters)?;
        let t_q8_dispatch = bench_quant_dispatch(shape, iters)?;

        let iters_f = iters as f64;
        println!("FP16 MLX GEMM:   total={:.3}s avg={:.6}ms", t_fp16, 1e3 * t_fp16 / iters_f);
        println!(
            "Q8  ctx.matmul:  total={:.3}s avg={:.6}ms",
            t_q8_dispatch,
            1e3 * t_q8_dispatch / iters_f
        );
    }
    Ok(())
}

#[test]
#[serial]
#[ignore]
fn bench_nt_crossover_m1_to_m4() -> Result<(), MetalError> {
    let shapes = [
        MatmulShape {
            m: 1,
            n: 512,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n512_k896_tb1",
        },
        MatmulShape {
            m: 1,
            n: 896,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n896_k896_tb1",
        },
        MatmulShape {
            m: 1,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n1536_k896_tb1",
        },
        MatmulShape {
            m: 1,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n2048_k896_tb1",
        },
        MatmulShape {
            m: 1,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n4096_k896_tb1",
        },
        MatmulShape {
            m: 1,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m1_n9728_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 512,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n512_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 896,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n896_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n1536_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n2048_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n4096_k896_tb1",
        },
        MatmulShape {
            m: 2,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m2_n9728_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 512,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n512_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 896,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n896_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n1536_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n2048_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n4096_k896_tb1",
        },
        MatmulShape {
            m: 3,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m3_n9728_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 512,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n512_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 896,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n896_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n1536_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n2048_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n4096_k896_tb1",
        },
        MatmulShape {
            m: 4,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: true,
            name: "m4_n9728_k896_tb1",
        },
    ];

    for shape in shapes {
        let iters = if shape.n > 20_000 { 200 } else { 1000 };
        println!(
            "\n== NT Crossover {}: m={} n={} k={} tA={} tB={} (iters={})",
            shape.name, shape.m, shape.n, shape.k, shape.t_a as u8, shape.t_b as u8, iters
        );

        let t_fp16 = bench_mlx_gemm_fp16(shape, iters)?;

        // Direct benches for specific kernels, avoiding env-var routing
        let t_q8_smallm_kernel = bench_q8_smallm_kernel_direct(shape, iters)?;
        let t_q8_canonical = if shape.m > 1 && shape.n >= 512 {
            Some(bench_q8_canonical_direct(shape, iters)?)
        } else {
            None
        };
        let t_q8_canonical_r16 = if shape.m > 1 && shape.n >= 512 {
            Some(bench_q8_canonical_rows16_direct(shape, iters)?)
        } else {
            None
        };
        let t_q8_direct = if shape.m == 1 {
            Some(bench_q8_direct_gemv(shape, iters)?)
        } else {
            None
        };
        let t_q8_nt = bench_q8_nt_direct(shape, iters)?;
        let iters_f = iters as f64;
        println!("FP16 MLX GEMM:       total={:.3}s avg={:.6}ms", t_fp16, 1e3 * t_fp16 / iters_f);
        println!(
            "Q8  small-m kernel:     total={:.3}s avg={:.6}ms",
            t_q8_smallm_kernel,
            1e3 * t_q8_smallm_kernel / iters_f
        );
        if let Some(t_canonical) = t_q8_canonical {
            println!(
                "Q8  canonical kernel:  total={:.3}s avg={:.6}ms",
                t_canonical,
                1e3 * t_canonical / iters_f
            );
        }
        if let Some(t_canonical_r16) = t_q8_canonical_r16 {
            println!(
                "Q8  canonical r16:     total={:.3}s avg={:.6}ms",
                t_canonical_r16,
                1e3 * t_canonical_r16 / iters_f
            );
        }
        if let Some(t_gemv) = t_q8_direct {
            println!("Q8  direct GEMV:        total={:.3}s avg={:.6}ms", t_gemv, 1e3 * t_gemv / iters_f);
        }
        println!("Q8  forced GEMM-NT:  total={:.3}s avg={:.6}ms", t_q8_nt, 1e3 * t_q8_nt / iters_f);
    }

    Ok(())
}

#[test]
#[serial]
#[ignore]
fn bench_crossover_m1_to_m4() -> Result<(), MetalError> {
    // Measure transpose_b=false with small m to see GEMM vs GEMV crossover.
    let shapes = [
        MatmulShape {
            m: 1,
            n: 512,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n512_k896_tb0",
        },
        MatmulShape {
            m: 1,
            n: 896,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n896_k896_tb0",
        },
        MatmulShape {
            m: 1,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n1536_k896_tb0",
        },
        MatmulShape {
            m: 1,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n2048_k896_tb0",
        },
        MatmulShape {
            m: 1,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n4096_k896_tb0",
        },
        MatmulShape {
            m: 1,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m1_n9728_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 512,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n512_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 896,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n896_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n1536_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n2048_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n4096_k896_tb0",
        },
        MatmulShape {
            m: 2,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m2_n9728_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 512,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n512_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 896,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n896_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n1536_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n2048_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n4096_k896_tb0",
        },
        MatmulShape {
            m: 3,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m3_n9728_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 512,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n512_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 896,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n896_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 1536,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n1536_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 2048,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n2048_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 4096,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n4096_k896_tb0",
        },
        MatmulShape {
            m: 4,
            n: 9728,
            k: 896,
            t_a: false,
            t_b: false,
            name: "m4_n9728_k896_tb0",
        },
    ];

    for shape in shapes {
        let iters = if shape.n > 20_000 { 200 } else { 1000 };
        println!(
            "\n== Crossover {}: m={} n={} k={} tA={} tB={} (iters={})",
            shape.name, shape.m, shape.n, shape.k, shape.t_a as u8, shape.t_b as u8, iters
        );

        let t_fp16 = bench_mlx_gemm_fp16(shape, iters)?;

        // Direct benches for specific kernels, avoiding env-var routing
        let t_q8_smallm_kernel = bench_q8_smallm_kernel_direct(shape, iters)?;
        let t_q8_canonical = if shape.m > 1 && shape.n >= 512 {
            Some(bench_q8_canonical_direct(shape, iters)?)
        } else {
            None
        };
        let t_q8_canonical_r16 = if shape.m > 1 && shape.n >= 512 {
            Some(bench_q8_canonical_rows16_direct(shape, iters)?)
        } else {
            None
        };
        let t_q8_direct = if shape.m == 1 {
            Some(bench_q8_direct_gemv(shape, iters)?)
        } else {
            None
        };
        let t_q8_nt = bench_q8_nt_direct(shape, iters)?;
        let t_q8_gemm = bench_mlx_gemm_q8(shape, iters)?;
        let iters_f = iters as f64;
        println!("FP16 MLX GEMM:       total={:.3}s avg={:.6}ms", t_fp16, 1e3 * t_fp16 / iters_f);
        println!(
            "Q8 MLX GEMM:         total={:.3}s avg={:.6}ms",
            t_q8_gemm,
            1e3 * t_q8_gemm / iters_f
        );

        println!(
            "Q8  small-m kernel:     total={:.3}s avg={:.6}ms",
            t_q8_smallm_kernel,
            1e3 * t_q8_smallm_kernel / iters_f
        );
        if let Some(t_canonical) = t_q8_canonical {
            println!(
                "Q8  canonical kernel:  total={:.3}s avg={:.6}ms",
                t_canonical,
                1e3 * t_canonical / iters_f
            );
        }
        if let Some(t_canonical_r16) = t_q8_canonical_r16 {
            println!(
                "Q8  canonical r16:     total={:.3}s avg={:.6}ms",
                t_canonical_r16,
                1e3 * t_canonical_r16 / iters_f
            );
        }
        if let Some(t_gemv) = t_q8_direct {
            println!("Q8  direct GEMV:        total={:.3}s avg={:.6}ms", t_gemv, 1e3 * t_gemv / iters_f);
        }
        println!("Q8  forced GEMM-NT:  total={:.3}s avg={:.6}ms", t_q8_nt, 1e3 * t_q8_nt / iters_f);
    }

    Ok(())
}

#[test]
#[serial]
#[ignore]
fn bench_qwen25_decode_sequence() -> Result<(), MetalError> {
    let iters = 200;
    let t_fp16 = bench_decode_sequence_fp16(iters)?;
    let t_q8 = bench_decode_sequence_q8(iters)?;
    println!("Decode FP16: total={:.3}s avg_iter={:.6}ms", t_fp16, 1e3 * t_fp16 / (iters as f64));
    println!("Decode Q8 GEMV: total={:.3}s avg_iter={:.6}ms", t_q8, 1e3 * t_q8 / (iters as f64));
    Ok(())
}
