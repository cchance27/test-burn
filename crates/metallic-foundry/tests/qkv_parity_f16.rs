//! F16 QKV Parity Test
//! Tests the fused QKV kernel with F16 weights (no quantization) directly
//! using CompoundKernel, mirroring the working Q8 test pattern.

use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, compound::{CompoundKernel, Layout, stages::WarpLayoutStage}, metals::{
        qkv::{
            stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, step::FusedQkvArgs
        }, rmsnorm::stages::RmsNormComputeStage
    }, policy::f16::PolicyF16, storage::Pooled, tensor::{F16, Tensor, TensorInit}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use rand::Rng;

fn run_cpu_rmsnorm(input: &[f16], gamma: &[f16]) -> Vec<f16> {
    let k = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        let x_f = x.to_f32();
        sum_sq += x_f * x_f;
    }
    let inv_rms = 1.0 / (sum_sq / k as f32 + 1e-6).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .map(|(&x, &g)| f16::from_f32(x.to_f32() * inv_rms * g.to_f32()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_qkv(
    input_norm: &[f16],
    w_q: &[f16],
    w_k: &[f16],
    w_v: &[f16],
    b_q: &[f16],
    b_k: &[f16],
    b_v: &[f16],
    k_dim: usize,
    n_dim: usize,
    n_kv: usize,
) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let mut out_q = vec![f16::from_f32(0.0); n_dim];
    let mut out_k = vec![f16::from_f32(0.0); n_kv];
    let mut out_v = vec![f16::from_f32(0.0); n_kv];

    for i in 0..n_dim {
        let mut sum = 0.0f32;
        for j in 0..k_dim {
            sum += input_norm[j].to_f32() * w_q[i * k_dim + j].to_f32();
        }
        out_q[i] = f16::from_f32(sum + b_q[i].to_f32());
    }

    for i in 0..n_kv {
        let mut sum_k = 0.0f32;
        let mut sum_v = 0.0f32;
        for j in 0..k_dim {
            sum_k += input_norm[j].to_f32() * w_k[i * k_dim + j].to_f32();
            sum_v += input_norm[j].to_f32() * w_v[i * k_dim + j].to_f32();
        }
        out_k[i] = f16::from_f32(sum_k + b_k[i].to_f32());
        out_v[i] = f16::from_f32(sum_v + b_v[i].to_f32());
    }

    (out_q, out_k, out_v)
}

fn assert_close(a: &[f16], b: &[f16], name: &str) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        max_diff = max_diff.max(diff);
        assert!(diff < 0.1, "{} disparity at index {}: CPU={} GPU={} diff={}", name, i, x, y, diff);
    }
    println!("{} max diff: {}", name, max_diff);
}

#[test]
fn test_qkv_parity_f16() {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rand::rng();

    let k_dim = 1024;
    let n_dim = 1024;
    let n_kv = 128; // GQA

    // Generate random data
    let mut x_data = vec![f16::from_f32(0.0); k_dim];
    let mut gamma_data = vec![f16::from_f32(0.0); k_dim];
    let mut w_q_data = vec![f16::from_f32(0.0); n_dim * k_dim];
    let mut w_k_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut w_v_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut b_q_data = vec![f16::from_f32(0.0); n_dim];
    let mut b_k_data = vec![f16::from_f32(0.0); n_kv];
    let mut b_v_data = vec![f16::from_f32(0.0); n_kv];

    for i in 0..k_dim {
        x_data[i] = f16::from_f32(rng.random_range(-1.0..1.0));
        gamma_data[i] = f16::from_f32(rng.random_range(0.5..1.5));
    }
    for val in w_q_data.iter_mut().take(n_dim * k_dim) {
        *val = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for i in 0..(n_kv * k_dim) {
        w_k_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
        w_v_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for val in b_q_data.iter_mut().take(n_dim) {
        *val = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for i in 0..n_kv {
        b_k_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
        b_v_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
    }

    // CPU Reference
    let x_norm = run_cpu_rmsnorm(&x_data, &gamma_data);
    let (ref_q, ref_k, ref_v) = run_cpu_qkv(
        &x_norm, &w_q_data, &w_k_data, &w_v_data, &b_q_data, &b_k_data, &b_v_data, k_dim, n_dim, n_kv,
    );

    // GPU Setup - Create Tensors
    let x_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&x_data)).unwrap();
    let gamma_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_data)).unwrap();

    // For F16: weights are passed as-is (no quantization)
    // The PolicyF16 handles uchar* -> half* casting internally
    let wq_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim * k_dim], TensorInit::CopyFrom(&w_q_data)).unwrap();
    let wk_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&w_k_data)).unwrap();
    let wv_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&w_v_data)).unwrap();

    // For F16: scales are dummies (PolicyF16::load_scale returns 1.0)
    // We reuse the weight tensors as dummy scale buffers
    let sq_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();
    let sk_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();
    let sv_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();

    let b_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_q_data)).unwrap();
    let b_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_k_data)).unwrap();
    let b_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_v_data)).unwrap();

    let out_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::Uninitialized).unwrap();
    let out_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();
    let out_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();

    // Compile Kernel using F16 Policy
    let kernel = Arc::new(
        CompoundKernel::new("fused_qkv_rmsnorm_f16_test")
            .with_manual_output(true)
            .prologue(WarpLayoutStage::new(Layout::RowMajor).with_warps(8))
            // FusedQkvArgs buffer indices:
            // input=6, k_dim=7, gamma=18, epsilon=19
            .prologue(RmsNormComputeStage::new(6, 7, 19))
            .main(ParallelProjectStage::new(std::sync::Arc::new(PolicyF16)).with_norm("inv_rms"))
            .epilogue(MultiWarpReduceStage)
            .epilogue(MultiWriteOutputStage::new())
            .compile(),
    );

    // DUMP KERNEL SOURCE FOR DEBUGGING
    println!("=== Generated Metal Source ===");
    println!("{}", kernel.source());
    println!("=== End Metal Source ===");

    let args = FusedQkvArgs {
        w_q: TensorArg::from_tensor(&wq_tensor),
        s_q: Some(TensorArg::from_tensor(&sq_tensor)),
        w_k: TensorArg::from_tensor(&wk_tensor),
        s_k: Some(TensorArg::from_tensor(&sk_tensor)),
        w_v: TensorArg::from_tensor(&wv_tensor),
        s_v: Some(TensorArg::from_tensor(&sv_tensor)),
        input: TensorArg::from_tensor(&x_tensor),
        k_dim: k_dim as u32,
        n_dim: n_dim as u32,
        n_kv: n_kv as u32,
        weights_per_block: 32, // Ignored by PolicyF16 but still needed for buffer layout
        out_q: TensorArg::from_tensor(&out_q_tensor),
        out_k: TensorArg::from_tensor(&out_k_tensor),
        out_v: TensorArg::from_tensor(&out_v_tensor),
        b_q: TensorArg::from_tensor(&b_q_tensor),
        b_k: TensorArg::from_tensor(&b_k_tensor),
        b_v: TensorArg::from_tensor(&b_v_tensor),
        has_bias: 1,
        gamma: TensorArg::from_tensor(&gamma_tensor),
        epsilon: 1e-6,
    };

    let warps_per_tg = 8;
    let num_groups = n_dim.max(n_kv).div_ceil(warps_per_tg);
    let dispatch = DispatchConfig {
        grid: GridSize::d1(num_groups),
        group: ThreadgroupSize::d1(warps_per_tg * 32),
    };

    foundry.run(&kernel.bind_arc(args.clone(), dispatch)).unwrap();

    let gpu_q = out_q_tensor.to_vec(&foundry);
    let gpu_k = out_k_tensor.to_vec(&foundry);
    let gpu_v = out_v_tensor.to_vec(&foundry);

    println!("--- F16 Parity ---");
    assert_close(&ref_q, &gpu_q, "Q Output");
    assert_close(&ref_k, &gpu_k, "K Output");
    assert_close(&ref_v, &gpu_v, "V Output");
}

/// Test FusedQkv kernel in BATCHED mode (start_capture/end_capture)
/// This mirrors how CompiledModel::forward() executes kernels.
/// If this test hangs, it confirms the deadlock is in batched execution.
#[test]
fn test_qkv_parity_f16_batched() {
    eprintln!("=== Testing FusedQkv in BATCHED mode ===");

    let mut foundry = Foundry::new().unwrap();
    let mut rng = rand::rng();

    let k_dim = 896; // Match Qwen2.5-0.5B d_model
    let n_dim = 896;
    let n_kv = 128; // 2 kv_heads * 64 head_dim

    // Generate random data
    let mut x_data = vec![f16::from_f32(0.0); k_dim];
    let mut gamma_data = vec![f16::from_f32(0.0); k_dim];
    let mut w_q_data = vec![f16::from_f32(0.0); n_dim * k_dim];
    let mut w_k_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut w_v_data = vec![f16::from_f32(0.0); n_kv * k_dim];
    let mut b_q_data = vec![f16::from_f32(0.0); n_dim];
    let mut b_k_data = vec![f16::from_f32(0.0); n_kv];
    let mut b_v_data = vec![f16::from_f32(0.0); n_kv];

    for i in 0..k_dim {
        x_data[i] = f16::from_f32(rng.random_range(-1.0..1.0));
        gamma_data[i] = f16::from_f32(rng.random_range(0.5..1.5));
    }
    for val in w_q_data.iter_mut().take(n_dim * k_dim) {
        *val = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for i in 0..(n_kv * k_dim) {
        w_k_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
        w_v_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for val in b_q_data.iter_mut().take(n_dim) {
        *val = f16::from_f32(rng.random_range(-0.1..0.1));
    }
    for i in 0..n_kv {
        b_k_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
        b_v_data[i] = f16::from_f32(rng.random_range(-0.1..0.1));
    }

    // CPU Reference
    let x_norm = run_cpu_rmsnorm(&x_data, &gamma_data);
    let (ref_q, ref_k, ref_v) = run_cpu_qkv(
        &x_norm, &w_q_data, &w_k_data, &w_v_data, &b_q_data, &b_k_data, &b_v_data, k_dim, n_dim, n_kv,
    );

    // GPU Setup
    let x_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&x_data)).unwrap();
    let gamma_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![k_dim], TensorInit::CopyFrom(&gamma_data)).unwrap();
    let wq_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim * k_dim], TensorInit::CopyFrom(&w_q_data)).unwrap();
    let wk_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&w_k_data)).unwrap();
    let wv_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv * k_dim], TensorInit::CopyFrom(&w_v_data)).unwrap();
    let sq_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();
    let sk_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();
    let sv_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::from_f32(1.0)])).unwrap();
    let b_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::CopyFrom(&b_q_data)).unwrap();
    let b_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_k_data)).unwrap();
    let b_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::CopyFrom(&b_v_data)).unwrap();
    let out_q_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_dim], TensorInit::Uninitialized).unwrap();
    let out_k_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();
    let out_v_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![n_kv], TensorInit::Uninitialized).unwrap();

    let kernel = Arc::new(
        CompoundKernel::new("fused_qkv_rmsnorm_f16_batched_test")
            .with_manual_output(true)
            .prologue(WarpLayoutStage::new(Layout::RowMajor).with_warps(8))
            // FusedQkvArgs buffer indices:
            // input=6, k_dim=7, gamma=18, epsilon=19
            .prologue(RmsNormComputeStage::new(6, 7, 19))
            .main(ParallelProjectStage::new(std::sync::Arc::new(PolicyF16)).with_norm("inv_rms"))
            .epilogue(MultiWarpReduceStage)
            .epilogue(MultiWriteOutputStage::new())
            .compile(),
    );

    let args = FusedQkvArgs {
        w_q: TensorArg::from_tensor(&wq_tensor),
        s_q: Some(TensorArg::from_tensor(&sq_tensor)),
        w_k: TensorArg::from_tensor(&wk_tensor),
        s_k: Some(TensorArg::from_tensor(&sk_tensor)),
        w_v: TensorArg::from_tensor(&wv_tensor),
        s_v: Some(TensorArg::from_tensor(&sv_tensor)),
        input: TensorArg::from_tensor(&x_tensor),
        k_dim: k_dim as u32,
        n_dim: n_dim as u32,
        n_kv: n_kv as u32,
        weights_per_block: 32,
        out_q: TensorArg::from_tensor(&out_q_tensor),
        out_k: TensorArg::from_tensor(&out_k_tensor),
        out_v: TensorArg::from_tensor(&out_v_tensor),
        b_q: TensorArg::from_tensor(&b_q_tensor),
        b_k: TensorArg::from_tensor(&b_k_tensor),
        b_v: TensorArg::from_tensor(&b_v_tensor),
        has_bias: 1,
        gamma: TensorArg::from_tensor(&gamma_tensor),
        epsilon: 1e-6,
    };

    let warps_per_tg = 8;
    let num_groups = n_dim.max(n_kv).div_ceil(warps_per_tg);
    let dispatch = DispatchConfig {
        grid: GridSize::d1(num_groups),
        group: ThreadgroupSize::d1(warps_per_tg * 32),
    };

    eprintln!("Starting batched capture...");
    foundry.start_capture().unwrap();

    eprintln!("Dispatching FusedQkv kernel in batched mode...");
    foundry.dispatch(&kernel.bind_arc(args.clone(), dispatch), dispatch).unwrap();

    eprintln!("Ending capture and waiting for completion...");
    let buffer = foundry.end_capture().unwrap();
    buffer.wait_until_completed();
    eprintln!("Batched execution completed successfully!");

    let gpu_q = out_q_tensor.to_vec(&foundry);
    let gpu_k = out_k_tensor.to_vec(&foundry);
    let gpu_v = out_v_tensor.to_vec(&foundry);

    println!("--- F16 Batched Parity ---");
    assert_close(&ref_q, &gpu_q, "Q Output (Batched)");
    assert_close(&ref_k, &gpu_k, "K Output (Batched)");
    assert_close(&ref_v, &gpu_v, "V Output (Batched)");
}
