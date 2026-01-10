//! GEMV V2 vs Context<T> Parity Test
//!
//! Compares the new V2 Stage-based GEMV implementation against
//! Context<T>'s MatmulGemvOp to ensure numerical consistency.

use half::f16;
use metallic::{
    Context, F16Element, MetalError, compound::stages::Layout, foundry::{Foundry, storage::Pooled, tensor::Tensor as FoundryTensor}, kernels::matmul_gemv::MatmulGemvOp, metals::gemv::{GemvStrategy, GemvV2Args, get_gemv_v2_kernel_f16, warp_dispatch_config}, tensor::{F16, Tensor, TensorInit, TensorStorage, TensorType}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 0.01; // Match existing fp16 tolerance

fn compare_results(v2: &[f16], context: &[f16], name: &str) {
    assert_eq!(v2.len(), context.len(), "Size mismatch in {}", name);

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (a, b)) in v2.iter().zip(context.iter()).enumerate() {
        let diff = (a.to_f32() - b.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("{}: max_diff={:.6} at idx {}", name, max_diff, max_idx);

    assert!(
        max_diff < TOLERANCE,
        "{} mismatch: max_diff={} at idx {} (V2={}, Ctx={})",
        name,
        max_diff,
        max_idx,
        v2[max_idx].to_f32(),
        context[max_idx].to_f32()
    );
}

fn run_gemv_parity_test(k: usize, n: usize, with_bias: bool, layout: Layout) -> Result<(), MetalError> {
    let mut rng = rng();

    // Generate random test data
    let weights_data: Vec<f16> = (0..k * n).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let input_data: Vec<f16> = (0..k).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let bias_data: Vec<f16> = (0..n).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // --- V2 GEMV (Foundry) ---
    let mut foundry = Foundry::new()?;

    let weights_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k * n], TensorInit::CopyFrom(&weights_data))?;
    let input_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k], TensorInit::CopyFrom(&input_data))?;
    let output_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;
    let bias_v2 = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::CopyFrom(&bias_data))?;

    let args = GemvV2Args {
        weights: TensorArg::from_tensor(&weights_v2),
        scale_bytes: TensorArg::from_tensor(&weights_v2), // Dummy for F16
        input: TensorArg::from_tensor(&input_v2),
        output: TensorArg::from_tensor(&output_v2),
        bias: if with_bias {
            TensorArg::from_tensor(&bias_v2)
        } else {
            TensorArg::from_tensor(&output_v2)
        },
        has_bias: if with_bias { 1 } else { 0 },
        k_dim: k as u32,
        n_dim: n as u32,
        weights_per_block: 32,
        alpha: 1.0,
        residual: TensorArg::from_tensor(&output_v2),
        has_residual: 0,
        beta: 0.0,
    };

    let kernel = get_gemv_v2_kernel_f16(layout, GemvStrategy::Vectorized);
    let dispatch = warp_dispatch_config(n as u32);
    foundry.run(&kernel.bind(args, dispatch))?;

    let res_v2 = output_v2.to_vec(&foundry);

    // --- Context GEMV ---
    let mut ctx = Context::<F16Element>::new()?;

    // Context expects [1, K] input shape for GEMV
    let input_ctx = Tensor::new(vec![1, k], TensorStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&input_data))?;

    // V2 Layout vs Context transpose_b mapping:
    // - Layout::RowMajor -> weights are [N, K] -> use Context transpose_b=true
    // - Layout::ColMajor -> weights are [K, N] -> use Context transpose_b=false
    let (weight_shape, transpose) = match layout {
        Layout::RowMajor => (vec![n, k], true),  // [N, K] with transpose_b=true
        Layout::ColMajor => (vec![k, n], false), // [K, N] with transpose_b=false
        Layout::Canonical => {
            // Canonical uses [N, K] like RowMajor
            (vec![n, k], true)
        }
    };

    let weights_ctx = Tensor::new(weight_shape, TensorStorage::Pooled(&mut ctx), TensorInit::CopyFrom(&weights_data))?;

    let bias_ctx = if with_bias {
        Some(Tensor::new(
            vec![n],
            TensorStorage::Pooled(&mut ctx),
            TensorInit::CopyFrom(&bias_data),
        )?)
    } else {
        None
    };

    let output_ctx = ctx.call::<MatmulGemvOp>((&input_ctx, TensorType::Dense(&weights_ctx), transpose, bias_ctx.as_ref()), None)?;

    ctx.synchronize();
    let res_ctx = output_ctx.try_to_vec()?;

    // Compare
    let label = format!("GEMV K={} N={} bias={} layout={:?}", k, n, with_bias, layout);
    compare_results(&res_v2, &res_ctx, &label);

    Ok(())
}

#[test]
#[serial]
fn test_gemv_v2_context_parity_row_major() -> Result<(), MetalError> {
    // Small
    run_gemv_parity_test(64, 64, false, Layout::RowMajor)?;
    run_gemv_parity_test(64, 64, true, Layout::RowMajor)?;
    // Medium
    run_gemv_parity_test(256, 256, false, Layout::RowMajor)?;
    run_gemv_parity_test(256, 256, true, Layout::RowMajor)?;
    Ok(())
}

#[test]
#[serial]
fn test_gemv_v2_context_parity_col_major() -> Result<(), MetalError> {
    // Small
    run_gemv_parity_test(64, 64, false, Layout::ColMajor)?;
    run_gemv_parity_test(64, 64, true, Layout::ColMajor)?;
    // Medium
    run_gemv_parity_test(256, 256, false, Layout::ColMajor)?;
    Ok(())
}

#[test]
#[serial]
fn test_gemv_v2_context_parity_large() -> Result<(), MetalError> {
    // Large shape (Qwen2.5 style)
    run_gemv_parity_test(896, 896, false, Layout::RowMajor)?;
    run_gemv_parity_test(896, 4864, false, Layout::RowMajor)?;
    Ok(())
}
