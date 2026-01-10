//! SDPA Materialized vs Context Parity Test
//!
//! Tests that materialized SDPA (GEMV + Softmax + GEMV) matches Context's SDPA exactly.
//! This should achieve perfect parity since both use the same algorithm.

use half::f16;
use metallic::{
    Context, F16Element, MetalError, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, tensor::{F16, Tensor, TensorInit, TensorStorage as LegacyStorage, dtypes::F16 as F16Type}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 0.01; // Should be very close with same algorithm

fn generate_random_f16(size: usize) -> Vec<f16> {
    let mut rng = rng();
    (0..size).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect()
}

fn compare_tensors(a: &[f16], b: &[f16], name: &str, tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Size mismatch");

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("{}: max_diff={:.6} at idx {}", name, max_diff, max_idx);

    // Print first failing details
    if max_diff >= tolerance {
        println!(
            "Details at idx {}: V2={:.6}, Ctx={:.6}, Diff={:.6}",
            max_idx,
            a[max_idx].to_f32(),
            b[max_idx].to_f32(),
            max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "{} mismatch: max_diff={} at idx {} (V2={}, Ctx={})",
        name,
        max_diff,
        max_idx,
        a[max_idx].to_f32(),
        b[max_idx].to_f32()
    );
}

/// CPU implementation of materialized SDPA for reference
fn cpu_materialized_sdpa(q: &[f16], k: &[f16], v: &[f16], batch: usize, q_len: usize, kv_len: usize, head_dim: usize) -> Vec<f16> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![f16::ZERO; batch * q_len * head_dim];

    for b in 0..batch {
        for qi in 0..q_len {
            // 1. Compute Q @ K^T (scaled)
            let mut scores = vec![0.0f32; kv_len];
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[b * q_len * head_dim + qi * head_dim + d].to_f32();
                    let k_val = k[b * kv_len * head_dim + ki * head_dim + d].to_f32();
                    dot += q_val * k_val;
                }
                scores[ki] = dot * scale;
            }

            // 2. Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let attn: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // 3. Attn @ V
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for ki in 0..kv_len {
                    let v_val = v[b * kv_len * head_dim + ki * head_dim + d].to_f32();
                    acc += attn[ki] * v_val;
                }
                output[b * q_len * head_dim + qi * head_dim + d] = f16::from_f32(acc);
            }
        }
    }
    output
}

/// Test materialized SDPA (via Context) vs CPU reference
#[test]
#[serial]
fn test_materialized_sdpa_vs_cpu() -> Result<(), MetalError> {
    let batch = 4;
    let q_len = 1;
    let kv_len = 64;
    let head_dim = 64;

    let q_data = generate_random_f16(batch * q_len * head_dim);
    let k_data = generate_random_f16(batch * kv_len * head_dim);
    let v_data = generate_random_f16(batch * kv_len * head_dim);

    let res_cpu = cpu_materialized_sdpa(&q_data, &k_data, &v_data, batch, q_len, kv_len, head_dim);

    let mut ctx = Context::<F16Element>::new()?;

    let q_ctx = Tensor::<F16>::new(
        vec![batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&q_data),
    )?;
    let k_ctx = Tensor::<F16>::new(
        vec![batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&k_data),
    )?;
    let v_ctx = Tensor::<F16>::new(
        vec![batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&v_data),
    )?;

    let out_ctx = ctx.scaled_dot_product_attention(&q_ctx, &k_ctx, &v_ctx, false)?;

    ctx.synchronize();
    let res_ctx = out_ctx.try_to_vec()?;

    compare_tensors(&res_ctx, &res_cpu, "Context SDPA vs CPU Materialized", TOLERANCE);

    Ok(())
}

/// Test that Context SDPA works correctly (baseline)
#[test]
#[serial]
fn test_context_sdpa_basic() -> Result<(), MetalError> {
    let batch = 2;
    let q_len = 1;
    let kv_len = 32;
    let head_dim = 64;

    let q_data = generate_random_f16(batch * q_len * head_dim);
    let k_data = generate_random_f16(batch * kv_len * head_dim);
    let v_data = generate_random_f16(batch * kv_len * head_dim);

    let mut ctx = Context::<F16Element>::new()?;

    let q_ctx = Tensor::<F16>::new(
        vec![batch, q_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&q_data),
    )?;
    let k_ctx = Tensor::<F16>::new(
        vec![batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&k_data),
    )?;
    let v_ctx = Tensor::<F16>::new(
        vec![batch, kv_len, head_dim],
        LegacyStorage::Pooled(&mut ctx),
        TensorInit::CopyFrom(&v_data),
    )?;

    let out_ctx = ctx.scaled_dot_product_attention(&q_ctx, &k_ctx, &v_ctx, false)?;

    ctx.synchronize();
    let res_ctx = out_ctx.try_to_vec()?;

    let res_cpu = cpu_materialized_sdpa(&q_data, &k_data, &v_data, batch, q_len, kv_len, head_dim);

    compare_tensors(&res_ctx, &res_cpu, "Context SDPA Basic", TOLERANCE);

    Ok(())
}

/// Test V2 materialized SDPA: GEMV(Q@K^T) + Softmax + GEMV(attn@V)
#[test]
#[serial]
fn test_v2_materialized_sdpa_decode() -> Result<(), MetalError> {
    use metallic::{
        compound::stages::Layout, metals::{
            gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel_f16, warp_dispatch_config}, softmax::{SoftmaxV2Args, get_softmax_v2_kernel}
        }, types::{DispatchConfig, GridSize, ThreadgroupSize}
    };

    let batch = 1; // Single batch for decode
    let q_len = 1;
    let kv_len = 64;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_data = generate_random_f16(batch * q_len * head_dim);
    let k_data = generate_random_f16(batch * kv_len * head_dim);
    let v_data = generate_random_f16(batch * kv_len * head_dim);

    // CPU reference
    let res_cpu = cpu_materialized_sdpa(&q_data, &k_data, &v_data, batch, q_len, kv_len, head_dim);

    // V2 Materialized: GEMV + Softmax + GEMV
    let mut foundry = Foundry::new()?;

    // Tensors
    let q_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![head_dim], TensorInit::CopyFrom(&q_data))?;
    let k_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len * head_dim], TensorInit::CopyFrom(&k_data))?;
    let v_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len * head_dim], TensorInit::CopyFrom(&v_data))?;

    // Intermediate buffers
    let attn_scores = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len], TensorInit::Uninitialized)?;
    let attn_probs = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![kv_len], TensorInit::Uninitialized)?;
    let out_v2 = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![head_dim], TensorInit::Uninitialized)?;
    let scale_tensor = FoundryTensor::<F16Type, Pooled>::new(&mut foundry, vec![1], TensorInit::CopyFrom(&[f16::ONE]))?;

    // Step 1: Q @ K^T
    let qk_args = GemvV2Args {
        weights: TensorArg::from_tensor(&k_v2),
        scale_bytes: TensorArg::from_tensor(&k_v2),
        input: TensorArg::from_tensor(&q_v2),
        output: TensorArg::from_tensor(&attn_scores),
        bias: TensorArg::from_tensor(&attn_scores),
        has_bias: 0,
        k_dim: head_dim as u32,
        n_dim: kv_len as u32,
        weights_per_block: 32,
        alpha: scale,
        residual: TensorArg::from_tensor(&attn_scores),
        has_residual: 0,
        beta: 0.0,
    };
    let qk_kernel = get_gemv_v2_kernel_f16(Layout::RowMajor, GemvStrategy::Vectorized);
    let qk_dispatch = warp_dispatch_config(kv_len as u32);
    foundry.run(&qk_kernel.bind(qk_args, qk_dispatch))?;

    // Step 2: Softmax(scores)
    let softmax_args = SoftmaxV2Args {
        input: TensorArg::from_tensor(&attn_scores),
        scale: TensorArg::from_tensor(&scale_tensor),
        output: TensorArg::from_tensor(&attn_probs),
        seq_k: kv_len as u32,
        causal: 0,
        query_offset: 0,
    };
    let softmax_dispatch = DispatchConfig {
        grid: GridSize::d2(1, 1),
        group: ThreadgroupSize::d1(256),
    };
    foundry.run(&get_softmax_v2_kernel().bind(softmax_args, softmax_dispatch))?;

    // Step 3: Probs @ V
    let av_args = GemvV2Args {
        weights: TensorArg::from_tensor(&v_v2),
        scale_bytes: TensorArg::from_tensor(&v_v2),
        input: TensorArg::from_tensor(&attn_probs),
        output: TensorArg::from_tensor(&out_v2),
        bias: TensorArg::from_tensor(&out_v2),
        has_bias: 0,
        k_dim: kv_len as u32,
        n_dim: head_dim as u32,
        weights_per_block: 32,
        alpha: 1.0,
        residual: TensorArg::from_tensor(&out_v2),
        has_residual: 0,
        beta: 0.0,
    };
    let av_kernel = get_gemv_v2_kernel_f16(Layout::ColMajor, GemvStrategy::Vectorized);
    let av_dispatch = warp_dispatch_config(head_dim as u32);
    foundry.run(&av_kernel.bind(av_args, av_dispatch))?;

    // Compare
    let res_v2 = out_v2.to_vec(&foundry);
    compare_tensors(&res_v2, &res_cpu, "V2 Materialized SDPA", TOLERANCE);

    Ok(())
}
