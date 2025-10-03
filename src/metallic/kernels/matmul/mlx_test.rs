use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};
use std::sync::{Mutex, OnceLock};

const FORCE_MATMUL_BACKEND_ENV: &str = "FORCE_MATMUL_BACKEND";

fn backend_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn set_backend_env_var(value: &str) {
    // SAFETY: Environment mutations are guarded by `backend_env_lock`, ensuring
    // there are no concurrent callers when we modify the process environment.
    unsafe {
        std::env::set_var(FORCE_MATMUL_BACKEND_ENV, value);
    }
}

fn restore_backend_env_var(previous: Option<String>) {
    // SAFETY: See `set_backend_env_var` for synchronization details.
    unsafe {
        if let Some(prev) = previous {
            std::env::set_var(FORCE_MATMUL_BACKEND_ENV, prev);
        } else {
            std::env::remove_var(FORCE_MATMUL_BACKEND_ENV);
        }
    }
}

fn new_context_for_backend(backend: &str) -> Result<Context<F32Element>, MetalError> {
    let guard = backend_env_lock().lock().expect("env mutex poisoned");
    let previous = std::env::var(FORCE_MATMUL_BACKEND_ENV).ok();
    set_backend_env_var(backend);
    let ctx_result = Context::<F32Element>::new();
    restore_backend_env_var(previous);
    drop(guard);
    ctx_result
}

fn tensor_from_data(ctx: &Context<F32Element>, dims: Vec<usize>, data: &[f32]) -> Result<Tensor<F32Element>, MetalError> {
    Tensor::new(dims, TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(data))
}

fn assert_tensors_close(lhs: &Tensor<F32Element>, rhs: &Tensor<F32Element>, tolerance: f32, msg: &str) {
    let lhs_vals = lhs.to_vec();
    let rhs_vals = rhs.to_vec();
    assert_eq!(lhs_vals.len(), rhs_vals.len(), "{} length mismatch", msg);

    for (idx, (l, r)) in lhs_vals.iter().zip(rhs_vals.iter()).enumerate() {
        let diff = (l - r).abs();
        assert!(
            diff <= tolerance,
            "{} mismatch at index {}: lhs={} rhs={} diff={} (> {})",
            msg,
            idx,
            l,
            r,
            diff,
            tolerance
        );
    }
}

#[test]
fn test_mlx_kernel_matches_mps_basic_matmul() -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let m = 5;
    let k = 7;
    let n = 3;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.031 + 0.5).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.017 + 0.25).collect();

    let a_mps = tensor_from_data(&ctx_mps, vec![m, k], &a_data)?;
    let b_mps = tensor_from_data(&ctx_mps, vec![k, n], &b_data)?;
    let a_mlx = tensor_from_data(&ctx_mlx, vec![m, k], &a_data)?;
    let b_mlx = tensor_from_data(&ctx_mlx, vec![k, n], &b_data)?;

    let out_mps = ctx_mps.matmul(&a_mps, &b_mps, false, false)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul(&a_mlx, &b_mlx, false, false)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-4, "basic matmul");
    Ok(())
}

#[test]
fn test_mlx_kernel_matches_mps_with_transpositions() -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    // Scenario 1: transpose left operand only.
    let m1 = 4;
    let k1 = 5;
    let n1 = 6;
    let a_left_data: Vec<f32> = (0..(k1 * m1)).map(|i| (i as f32) * 0.07 - 0.15).collect();
    let b_left_data: Vec<f32> = (0..(k1 * n1)).map(|i| (i as f32) * 0.02 + 0.45).collect();

    let a_left_mps = tensor_from_data(&ctx_mps, vec![k1, m1], &a_left_data)?;
    let b_left_mps = tensor_from_data(&ctx_mps, vec![k1, n1], &b_left_data)?;
    let a_left_mlx = tensor_from_data(&ctx_mlx, vec![k1, m1], &a_left_data)?;
    let b_left_mlx = tensor_from_data(&ctx_mlx, vec![k1, n1], &b_left_data)?;

    let out_left_mps = ctx_mps.matmul(&a_left_mps, &b_left_mps, true, false)?;
    ctx_mps.synchronize();
    let out_left_mlx = ctx_mlx.matmul(&a_left_mlx, &b_left_mlx, true, false)?;
    ctx_mlx.synchronize();
    assert_tensors_close(&out_left_mps, &out_left_mlx, 1e-4, "transpose-left matmul");

    // Scenario 2: transpose right operand only.
    let m2 = 3;
    let k2 = 4;
    let n2 = 7;
    let a_right_data: Vec<f32> = (0..(m2 * k2)).map(|i| (i as f32) * -0.09 + 0.6).collect();
    let b_right_data: Vec<f32> = (0..(n2 * k2)).map(|i| (i as f32) * 0.05 - 0.35).collect();

    let a_right_mps = tensor_from_data(&ctx_mps, vec![m2, k2], &a_right_data)?;
    let b_right_mps = tensor_from_data(&ctx_mps, vec![n2, k2], &b_right_data)?;
    let a_right_mlx = tensor_from_data(&ctx_mlx, vec![m2, k2], &a_right_data)?;
    let b_right_mlx = tensor_from_data(&ctx_mlx, vec![n2, k2], &b_right_data)?;

    let out_right_mps = ctx_mps.matmul(&a_right_mps, &b_right_mps, false, true)?;
    ctx_mps.synchronize();
    let out_right_mlx = ctx_mlx.matmul(&a_right_mlx, &b_right_mlx, false, true)?;
    ctx_mlx.synchronize();
    assert_tensors_close(&out_right_mps, &out_right_mlx, 1e-4, "transpose-right matmul");

    // Scenario 3: transpose both operands.
    let m3 = 2;
    let k3 = 5;
    let n3 = 4;
    let a_both_data: Vec<f32> = (0..(k3 * m3)).map(|i| (i as f32) * 0.013 + 0.21).collect();
    let b_both_data: Vec<f32> = (0..(n3 * k3)).map(|i| (i as f32) * -0.025 + 0.12).collect();

    let a_both_mps = tensor_from_data(&ctx_mps, vec![k3, m3], &a_both_data)?;
    let b_both_mps = tensor_from_data(&ctx_mps, vec![n3, k3], &b_both_data)?;
    let a_both_mlx = tensor_from_data(&ctx_mlx, vec![k3, m3], &a_both_data)?;
    let b_both_mlx = tensor_from_data(&ctx_mlx, vec![n3, k3], &b_both_data)?;

    let out_both_mps = ctx_mps.matmul(&a_both_mps, &b_both_mps, true, true)?;
    ctx_mps.synchronize();
    let out_both_mlx = ctx_mlx.matmul(&a_both_mlx, &b_both_mlx, true, true)?;
    ctx_mlx.synchronize();
    assert_tensors_close(&out_both_mps, &out_both_mlx, 1e-4, "transpose-both matmul");

    Ok(())
}

#[test]
fn test_mlx_kernel_matches_mps_alpha_beta() -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let m = 3;
    let k = 4;
    let n = 5;
    let alpha = 0.75;
    let beta = -1.25;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.04 + 0.3).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.03 + 0.6).collect();
    let c_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.015 - 0.1).collect();

    let a_mps = tensor_from_data(&ctx_mps, vec![m, k], &a_data)?;
    let b_mps = tensor_from_data(&ctx_mps, vec![k, n], &b_data)?;
    let c_mps = tensor_from_data(&ctx_mps, vec![m, n], &c_data)?;

    let a_mlx = tensor_from_data(&ctx_mlx, vec![m, k], &a_data)?;
    let b_mlx = tensor_from_data(&ctx_mlx, vec![k, n], &b_data)?;
    let c_mlx = tensor_from_data(&ctx_mlx, vec![m, n], &c_data)?;

    let out_mps = ctx_mps.matmul_alpha_beta(&a_mps, &b_mps, &c_mps, false, false, alpha, beta)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul_alpha_beta(&a_mlx, &b_mlx, &c_mlx, false, false, alpha, beta)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-4, "alpha-beta matmul");
    Ok(())
}

#[test]
fn test_mlx_kernel_matches_mps_non_contiguous_views() -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let m = 3;
    let k = 4;
    let n = 5;
    let padded_k = k + 2;
    let padded_n = n + 3;

    let a_padded_data: Vec<f32> = (0..(m * padded_k)).map(|i| (i as f32) * 0.01 + 0.2).collect();
    let b_padded_data: Vec<f32> = (0..(k * padded_n)).map(|i| (i as f32) * -0.02 + 0.35).collect();

    let a_padded_mps = tensor_from_data(&ctx_mps, vec![m, padded_k], &a_padded_data)?;
    let b_padded_mps = tensor_from_data(&ctx_mps, vec![k, padded_n], &b_padded_data)?;
    let a_view_mps = a_padded_mps.slice_last_dim(0..k)?;
    let b_view_mps = b_padded_mps.slice_last_dim(0..n)?;

    let a_padded_mlx = tensor_from_data(&ctx_mlx, vec![m, padded_k], &a_padded_data)?;
    let b_padded_mlx = tensor_from_data(&ctx_mlx, vec![k, padded_n], &b_padded_data)?;
    let a_view_mlx = a_padded_mlx.slice_last_dim(0..k)?;
    let b_view_mlx = b_padded_mlx.slice_last_dim(0..n)?;

    let out_mps = ctx_mps.matmul(&a_view_mps, &b_view_mps, false, false)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul(&a_view_mlx, &b_view_mlx, false, false)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-4, "non-contiguous matmul");
    Ok(())
}

#[test]
fn test_mlx_kernel_matches_mps_batched_matmul() -> Result<(), MetalError> {
    let mut ctx_mps = new_context_for_backend("mps")?;
    let mut ctx_mlx = new_context_for_backend("mlx")?;

    let batch = 3;
    let m = 2;
    let k = 5;
    let n = 4;

    let a_data: Vec<f32> = (0..(batch * m * k)).map(|i| (i as f32) * 0.011 + 0.4).collect();
    let b_data: Vec<f32> = (0..(batch * k * n)).map(|i| (i as f32) * -0.023 + 0.18).collect();

    let a_mps = tensor_from_data(&ctx_mps, vec![batch, m, k], &a_data)?;
    let b_mps = tensor_from_data(&ctx_mps, vec![batch, k, n], &b_data)?;
    let a_mlx = tensor_from_data(&ctx_mlx, vec![batch, m, k], &a_data)?;
    let b_mlx = tensor_from_data(&ctx_mlx, vec![batch, k, n], &b_data)?;

    let out_mps = ctx_mps.matmul(&a_mps, &b_mps, false, false)?;
    ctx_mps.synchronize();
    let out_mlx = ctx_mlx.matmul(&a_mlx, &b_mlx, false, false)?;
    ctx_mlx.synchronize();

    assert_tensors_close(&out_mps, &out_mlx, 1e-4, "batched matmul");
    Ok(())
}
