use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, compound::{
        CompoundKernel, Layout, stages::{WarpLayoutStage, WarpReduceStage}
    }, metals::{
        gemv::{
            fused_step::FusedGemvArgs, stages::{VectorizedDotStage, WarpWriteOutputNoResidualStage}, warp_dispatch_config
        }, rmsnorm::stages::RmsNormComputeStage
    }, policy::q8::PolicyQ8, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit, U8}, types::TensorArg
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serial_test::serial;

// ============================================================================
// CPU Reference Helpers
// ============================================================================

fn run_cpu_rmsnorm(input: &[f16], gamma: &[f16]) -> Vec<f16> {
    let k = input.len();
    let sum_sq: f32 = input
        .iter()
        .map(|x| {
            let val = x.to_f32();
            val * val
        })
        .sum();
    let mean_sq = sum_sq / (k as f32);
    let inv_rms = (mean_sq + 1e-6).sqrt().recip();

    input
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| {
            let val = x.to_f32() * inv_rms * g.to_f32();
            f16::from_f32(val)
        })
        .collect()
}

// Minimal Q8 GEMV CPU Ref
#[allow(clippy::too_many_arguments)]
fn run_cpu_gemv_q8(
    k: usize,
    n: usize,
    layout: Layout,
    weights_i8: &[i8],
    scales: &[f16],
    weights_per_block: usize,
    input: &[f16],
    bias: Option<&[f16]>,
    alpha: f32,
) -> Vec<f16> {
    let mut output = vec![f16::from_f32(0.0); n];
    let blocks_per_k = k.div_ceil(weights_per_block);
    for row in 0..n {
        let mut acc = 0.0f32;
        let mut ki = 0;
        while ki < k {
            let chunk_size = if ki + 8 <= k { 8 } else { k - ki };
            let mut chunk_acc = 0.0f32;
            let block_idx = ki / weights_per_block;
            let scale_idx = row * blocks_per_k + block_idx;
            let scale = scales[scale_idx].to_f32();

            for i in 0..chunk_size {
                let curr_k = ki + i;
                let w_idx = match layout {
                    Layout::RowMajor => row * k + curr_k,
                    Layout::ColMajor => curr_k * n + row,
                    _ => row * k + curr_k,
                };
                let w = weights_i8[w_idx] as f32;
                let x = input[curr_k].to_f32();
                chunk_acc += w * x;
            }
            acc += chunk_acc * scale;
            ki += chunk_size;
        }
        let mut res = acc * alpha;
        if let Some(b) = bias {
            res += b[row].to_f32();
        }
        output[row] = f16::from_f32(res);
    }
    output
}

// ============================================================================
// Test Runner
// ============================================================================

#[derive(Clone)]
struct FusedTestConfig {
    k: usize,
    n: usize,
    with_bias: bool,
    with_norm: bool,
}

impl Default for FusedTestConfig {
    fn default() -> Self {
        Self {
            k: 256,
            n: 256,
            with_bias: false,
            with_norm: true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_fused_gemv_test(cfg: FusedTestConfig) {
    let mut foundry = Foundry::new().unwrap();
    let seed =
        (cfg.k as u64) ^ ((cfg.n as u64) << 32) ^ ((cfg.with_bias as u64) << 1) ^ ((cfg.with_norm as u64) << 2) ^ 0x9e37_79b9_7f4a_7c15;
    let mut rng = StdRng::seed_from_u64(seed);

    // Setup Data
    let block_size = 32;
    let n_blocks = (cfg.k * cfg.n) / block_size;

    let weights_data: Vec<u8> = (0..cfg.k * cfg.n).map(|_| rng.random_range(0..255)).collect();
    let scales_data: Vec<f16> = (0..n_blocks).map(|_| f16::from_f32(rng.random_range(0.5..1.5))).collect();
    let input_data: Vec<f16> = (0..cfg.k).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let gamma_data: Vec<f16> = (0..cfg.k).map(|_| f16::from_f32(rng.random_range(0.8..1.2))).collect();
    let bias_data: Vec<f16> = (0..cfg.n).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Create Foundry Tensors
    let weights = FoundryTensor::<U8, Pooled>::new(&mut foundry, vec![cfg.k * cfg.n], TensorInit::CopyFrom(&weights_data)).unwrap();
    let scales = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_blocks], TensorInit::CopyFrom(&scales_data)).unwrap();
    let input = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&input_data)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::Uninitialized).unwrap();
    let gamma = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.k], TensorInit::CopyFrom(&gamma_data)).unwrap();
    let bias = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![cfg.n], TensorInit::CopyFrom(&bias_data)).unwrap();

    // 1. Run CPU Reference
    let norm_input = if cfg.with_norm {
        run_cpu_rmsnorm(&input_data, &gamma_data)
    } else {
        input_data.clone()
    };

    let weights_i8: Vec<i8> = weights_data.iter().map(|&x| x as i8).collect();
    let cpu_output_f16 = run_cpu_gemv_q8(
        cfg.k,
        cfg.n,
        Layout::RowMajor,
        &weights_i8,
        &scales_data,
        block_size,
        &norm_input,
        if cfg.with_bias { Some(&bias_data) } else { None },
        1.0,
    );
    let cpu_output: Vec<f32> = cpu_output_f16.iter().map(|x| x.to_f32()).collect();

    // 2. Run Fused Kernel
    let mut builder = CompoundKernel::new("test_fused_gemv")
        .with_manual_output(true)
        .prologue(WarpLayoutStage::new(Layout::RowMajor).with_warps(8)); // Defines row_idx, lane_id

    if cfg.with_norm {
        builder = builder.prologue(RmsNormComputeStage::new(2, 4, 11));
    }

    let mut dot_stage = VectorizedDotStage::new(std::sync::Arc::new(PolicyQ8));
    if cfg.with_norm {
        dot_stage = dot_stage.with_norm(10, "inv_rms");
    }

    let kernel = Arc::new(
        builder
            .main(dot_stage)
            .epilogue(WarpReduceStage::sum("partial_dot", "row_sum"))
            .epilogue(WarpWriteOutputNoResidualStage::new())
            .compile(),
    );

    let args = FusedGemvArgs {
        weights: TensorArg::from_tensor(&weights),
        scales: Some(TensorArg::from_tensor(&scales)),
        input: TensorArg::from_tensor(&input),
        output: TensorArg::from_tensor(&output),
        k_dim: cfg.k as u32,
        n_dim: cfg.n as u32,
        weights_per_block: 32,
        bias: if cfg.with_bias {
            TensorArg::from_tensor(&bias)
        } else {
            TensorArg::from_tensor(&output)
        },
        has_bias: if cfg.with_bias { 1 } else { 0 },
        alpha: 1.0,
        gamma: if cfg.with_norm {
            TensorArg::from_tensor(&gamma)
        } else {
            TensorArg::from_tensor(&input)
        }, // Dummy if unused
        epsilon: 1e-6,
    };

    let dispatch = warp_dispatch_config(cfg.n as u32);

    foundry.run(&kernel.bind_arc(args, dispatch)).unwrap();

    // 3. Compare Results
    let v2_f16 = FoundryTensor::to_vec(&output, &foundry);
    let v2_output: Vec<f32> = v2_f16.iter().map(|x| x.to_f32()).collect();

    let tolerance = 5e-2 * (cfg.k as f32).sqrt();

    let mut max_diff = 0.0f32;
    for (l, v) in cpu_output.iter().zip(v2_output.iter()) {
        max_diff = max_diff.max((l - v).abs());
    }

    println!(
        "\n=== Fused Gemv Parity Config: {:?} ===",
        (cfg.k, cfg.n, cfg.with_bias, cfg.with_norm)
    );
    println!("Max diff: {}", max_diff);
    println!("Tolerance: {}", tolerance);

    if max_diff > tolerance {
        println!("CPU First 5: {:?}", &cpu_output[..5]);
        println!("GPU First 5: {:?}", &v2_output[..5]);
        panic!("Fused Gemv Parity failed!");
    }
}

#[test]
#[serial]
fn test_fused_gemv_basic() {
    run_fused_gemv_test(FusedTestConfig::default());
}

#[test]
#[serial]
fn test_plain_gemv() {
    run_fused_gemv_test(FusedTestConfig {
        with_norm: false,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_fused_gemv_with_bias() {
    run_fused_gemv_test(FusedTestConfig {
        with_bias: true,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_fused_gemv_large_k() {
    run_fused_gemv_test(FusedTestConfig {
        k: 1024,
        n: 128,
        ..Default::default()
    });
}

#[test]
#[serial]
fn test_fused_gemv_large_n() {
    run_fused_gemv_test(FusedTestConfig {
        k: 128,
        n: 1024,
        ..Default::default()
    });
}
