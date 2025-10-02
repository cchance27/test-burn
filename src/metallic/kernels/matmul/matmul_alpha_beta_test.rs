use super::*;

use super::matmul_alpha_beta;
use super::matmul_test::cpu_matmul_scaled;
use crate::metallic::{F32Element, TensorInit, TensorStorage};

#[test]
fn test_matmul_alpha_beta_accumulation() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix
    let c_data = vec![0.5, 1.5, 2.5, 3.5]; // 2x2 matrix (will be used as result with beta)

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?; // Will be used as result with beta

    let alpha = 0.5;
    let beta = 0.25;

    // Expected result: alpha * A * B + beta * C
    // A * B = [[31, 19], [85, 55]] (calculated manually)
    // alpha * A * B = [[15.5, 9.5], [42.5, 27.5]]
    // beta * C = [[0.125, 0.375], [0.625, 0.875]]
    // Final result = [[15.625, 9.875], [43.125, 28.375]]
    let expected_result = [15.625, 9.875, 43.125, 28.375];

    // Use the new kernel system with alpha/beta scaling
    let result_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val = expected_result[i];
        let diff = (metal_val - expected_val).abs();
        let rel_err = if expected_val.abs() > 1e-8 {
            diff / expected_val.abs()
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, expected={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            expected_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_with_different_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 3;
    let k = 2;
    let n = 4;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.5).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.3).collect();
    let c_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.1).collect();

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?;

    let alpha = 2.0;
    let beta = -0.5;

    // Calculate expected result manually
    // A = [[0, 0.5], [1, 1.5], [2, 2.5]]
    // B = [[0, 0.3, 0.6, 0.9], [1.2, 1.5, 1.8, 2.1]]
    // A * B = [[0.6, 0.75, 0.9, 1.05], [1.8, 2.55, 3.3, 4.05], [3.0, 4.35, 5.7, 7.05]]
    // alpha * A * B = [[1.2, 1.5, 1.8, 2.1], [3.6, 5.1, 6.6, 8.1], [6.0, 8.7, 11.4, 14.1]]
    // beta * C = [[0, -0.05, -0.1, -0.15], [-0.2, -0.25, -0.3, -0.35], [-0.4, -0.45, -0.5, -0.55]]
    // Final result = [[1.2, 1.45, 1.7, 1.95], [3.4, 4.85, 6.3, 7.75], [5.6, 8.25, 10.9, 13.55]]
    let expected_result = [1.2, 1.45, 1.7, 1.95, 3.4, 4.85, 6.3, 7.75, 5.6, 8.25, 10.9, 13.55];

    // Use the new kernel system with alpha/beta scaling
    let result_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val = expected_result[i];
        let diff = (metal_val - expected_val).abs();
        let rel_err = if expected_val.abs() > 1e-8 {
            diff / expected_val.abs()
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, expected={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            expected_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_zero_beta_one() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let c_data = vec![10.0, 20.0, 30.0, 40.0]; // This should be the result when alpha=0, beta=1

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?;

    let alpha = 0.0;
    let beta = 1.0;

    // Expected result: 0 * A * B + 1 * C = C
    let expected_result = c_data.clone();

    // Use the new kernel system with alpha/beta scaling
    let result_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val = expected_result[i] as f64;
        let diff = (metal_val - expected_val).abs();
        let rel_err = if expected_val.abs() > 1e-8 {
            diff / expected_val.abs()
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, expected={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            expected_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_one_beta_zero() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let c_data = vec![0.0, 0.0, 0.0, 0.0]; // This will be overwritten when alpha=1, beta=0

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?;

    let alpha = 1.0;
    let beta = 0.0;

    // Expected result: 1 * A * B + 0 * C = A * B
    // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected_result = [19.0, 22.0, 43.0, 50.0];

    // Use the new kernel system with alpha/beta scaling
    let result_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val = expected_result[i];
        let diff = (metal_val - expected_val).abs();
        let rel_err = if expected_val.abs() > 1e-8 {
            diff / expected_val.abs()
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, expected={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            expected_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

fn make_result_tensor(context: &Context<F32Element>, data: &[f32], dims: &[usize]) -> Result<Tensor<F32Element>, MetalError> {
    Tensor::new(dims.to_vec(), TensorStorage::Dedicated(context), TensorInit::CopyFrom(data))
}

fn verify_alpha_beta_backend(transpose_left: bool, transpose_right: bool) -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 4;
    let k = 3;
    let n = 5;

    let (left_rows, left_cols) = if transpose_left { (k, m) } else { (m, k) };
    let (right_rows, right_cols) = if transpose_right { (n, k) } else { (k, n) };

    let left_data: Vec<f32> = (0..(left_rows * left_cols)).map(|i| (i as f32) * 0.03125 + 0.125).collect();
    let right_data: Vec<f32> = (0..(right_rows * right_cols)).map(|i| 1.0 - (i as f32) * 0.0175).collect();
    let result_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32).sin() * 0.25).collect();

    let left_tensor = Tensor::new(
        vec![left_rows, left_cols],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&left_data),
    )?;
    let right_tensor = Tensor::new(
        vec![right_rows, right_cols],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&right_data),
    )?;

    let alpha = 0.75f32;
    let beta = 0.5f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &result_data, &[m, n])?;
    let mps_result = context.matmul_alpha_beta(
        &left_tensor,
        &right_tensor,
        &result_tensor_mps,
        transpose_left,
        transpose_right,
        alpha,
        beta,
    )?;
    context.synchronize();
    let expected = mps_result.to_vec();
    let mps_samples = context.take_matmul_samples();
    assert!(
        mps_samples.iter().all(|sample| sample.backend == MatMulBackend::Mps),
        "expected ForceMps dispatches to use the MPS backend"
    );

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &result_data, &[m, n])?;
    let mlx_result = context.matmul_alpha_beta(
        &left_tensor,
        &right_tensor,
        &result_tensor_mlx,
        transpose_left,
        transpose_right,
        alpha,
        beta,
    )?;
    context.synchronize();
    let actual = mlx_result.to_vec();
    let mlx_samples = context.take_matmul_samples();
    let expected_backend = if transpose_left || transpose_right {
        MatMulBackend::MlxTransposed
    } else {
        MatMulBackend::Mlx
    };
    assert!(
        mlx_samples.iter().all(|sample| sample.backend == expected_backend),
        "expected MLX backend {:?} but observed {:?}",
        expected_backend,
        mlx_samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    assert_eq!(expected.len(), actual.len());
    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - act).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} for transpose_left={}, transpose_right={} (expected {}, got {}, diff {}, rel {})",
            idx,
            transpose_left,
            transpose_right,
            exp,
            act,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_selects_mlx_backend() -> Result<(), MetalError> {
    verify_alpha_beta_backend(false, false)?;
    verify_alpha_beta_backend(true, false)?;
    verify_alpha_beta_backend(false, true)?;
    verify_alpha_beta_backend(true, true)?;
    Ok(())
}

#[test]
fn test_matmul_alpha_beta_batched_mlx_matches_mps() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 3;

    let a_data: Vec<f32> = (0..(batch * m * k)).map(|i| (i as f32) * 0.0125 + 0.1).collect();
    let b_data: Vec<f32> = (0..(batch * k * n)).map(|i| 0.9 - (i as f32) * 0.015).collect();
    let c_data: Vec<f32> = (0..(batch * m * n)).map(|i| (i as f32).cos() * 0.2).collect();

    let a_tensor = Tensor::new(vec![batch, m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![batch, k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let alpha = 1.25f32;
    let beta = -0.4f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &c_data, &[batch, m, n])?;
    let mps_result = context.matmul_alpha_beta(&a_tensor, &b_tensor, &result_tensor_mps, false, false, alpha, beta)?;
    context.synchronize();
    let expected = mps_result.to_vec();
    let mps_samples = context.take_matmul_samples();
    assert!(
        mps_samples.iter().all(|sample| sample.backend == MatMulBackend::Mps),
        "expected ForceMps dispatches to use the MPS backend"
    );

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &c_data, &[batch, m, n])?;
    let mlx_result = context.matmul_alpha_beta(&a_tensor, &b_tensor, &result_tensor_mlx, false, false, alpha, beta)?;
    context.synchronize();
    let actual = mlx_result.to_vec();
    let mlx_samples = context.take_matmul_samples();
    assert!(
        !mlx_samples.is_empty() && mlx_samples.iter().all(|sample| sample.backend == MatMulBackend::Mlx),
        "expected batched MLX dispatches to report the MLX backend"
    );

    assert_eq!(expected.len(), actual.len());
    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - act).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} in batched comparison (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            act,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_scale_only_avoids_output_reads() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 4;
    let k = 3;
    let n = 5;

    let left_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.0425 + 0.15).collect();
    let right_data: Vec<f32> = (0..(k * n)).map(|i| 1.0 - (i as f32) * 0.021).collect();
    let result_data: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.037).sin()).collect();

    let left_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&left_data))?;
    let right_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&right_data))?;

    let alpha = 0.5f32;
    let beta = 0.0f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &result_data, &[m, n])?;
    let expected_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor_mps, false, false, alpha, beta)?;
    context.synchronize();
    let expected = expected_tensor.to_vec();
    context.take_matmul_samples();

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &result_data, &[m, n])?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor_mlx, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();
    let mlx_samples = context.take_matmul_samples();
    let non_total_backends: Vec<MatMulBackend> = mlx_samples
        .iter()
        .map(|sample| sample.backend)
        .filter(|backend| *backend != MatMulBackend::Total)
        .collect();
    assert!(
        !non_total_backends.is_empty() && non_total_backends.iter().all(|backend| *backend == MatMulBackend::Mlx),
        "expected MLX backend but observed {:?}",
        mlx_samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let flags = matmul_alpha_beta::take_last_mlx_alpha_beta_flags().expect("expected MLX alpha/beta flags to be recorded");
    assert_eq!(flags, (false, true), "expected write-only destination in scale-only mode");

    assert_eq!(expected.len(), actual.len());
    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - act).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            act,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_batched_addmm_reads_output() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let batch = 3;
    let m = 2;
    let k = 4;
    let n = 3;

    let left_data: Vec<f32> = (0..(batch * m * k)).map(|i| (i as f32) * 0.021 + 0.35).collect();
    let right_data: Vec<f32> = (0..(batch * k * n)).map(|i| 1.1 - (i as f32) * 0.017).collect();
    let result_data: Vec<f32> = (0..(batch * m * n)).map(|i| ((i as f32) * 0.045).cos() * 0.3).collect();

    let left_tensor = Tensor::new(
        vec![batch, m, k],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&left_data),
    )?;
    let right_tensor = Tensor::new(
        vec![batch, k, n],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&right_data),
    )?;

    let alpha = 0.85f32;
    let beta = 0.25f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &result_data, &[batch, m, n])?;
    let expected_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor_mps, false, false, alpha, beta)?;
    context.synchronize();
    let expected = expected_tensor.to_vec();
    context.take_matmul_samples();

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &result_data, &[batch, m, n])?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor_mlx, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();
    let mlx_samples = context.take_matmul_samples();
    let non_total_backends: Vec<MatMulBackend> = mlx_samples
        .iter()
        .map(|sample| sample.backend)
        .filter(|backend| *backend != MatMulBackend::Total)
        .collect();
    assert!(
        !non_total_backends.is_empty() && non_total_backends.iter().all(|backend| *backend == MatMulBackend::Mlx),
        "expected MLX backend but observed {:?}",
        mlx_samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let flags = matmul_alpha_beta::take_last_mlx_alpha_beta_flags().expect("expected MLX alpha/beta flags to be recorded");
    assert_eq!(flags, (true, false), "expected output reads when beta is non-zero");

    assert_eq!(expected.len(), actual.len());
    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - act).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} in batched addmm comparison (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            act,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_accepts_strided_kv_view() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let batch_heads = 2;
    let cache_capacity = 6;
    let active_steps = 4;
    let query_len = 3;
    let head_dim = 5;

    let left_data: Vec<f32> = (0..(batch_heads * query_len * active_steps))
        .map(|i| (i as f32) * 0.0375 - 0.2)
        .collect();
    let cache_data: Vec<f32> = (0..(batch_heads * cache_capacity * head_dim))
        .map(|i| (i as f32) * 0.013 + 0.45)
        .collect();

    let left_tensor = Tensor::new(
        vec![batch_heads, query_len, active_steps],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&left_data),
    )?;
    let cache_tensor = Tensor::new(
        vec![batch_heads, cache_capacity, head_dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cache_data),
    )?;

    let (value_history_view, returned_capacity) = context.kv_cache_history_view(&cache_tensor, active_steps)?;
    assert_eq!(returned_capacity, cache_capacity);

    let padded_view = value_history_view.as_mps_matrix_batch_view()?;
    assert!(
        padded_view.matrix_bytes > padded_view.rows * padded_view.row_bytes,
        "expected KV cache view to report padded batch stride"
    );

    let result_dims = [batch_heads, query_len, head_dim];
    let result_data = vec![0.0f32; result_dims.iter().product()];
    // Use a non-unit alpha to ensure the MLX path exercises the scale-only fast path
    let alpha = 0.5f32;
    let beta = 0.0f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &result_data, &result_dims)?;
    let expected_tensor = context.matmul_alpha_beta(&left_tensor, &value_history_view, &result_tensor_mps, false, false, alpha, beta)?;
    context.synchronize();
    let expected = expected_tensor.to_vec();
    context.take_matmul_samples();

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &result_data, &result_dims)?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &value_history_view, &result_tensor_mlx, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();
    let mlx_samples = context.take_matmul_samples();
    let non_total_backends: Vec<MatMulBackend> = mlx_samples
        .iter()
        .map(|sample| sample.backend)
        .filter(|backend| *backend != MatMulBackend::Total)
        .collect();
    assert!(
        !non_total_backends.is_empty() && non_total_backends.iter().all(|backend| *backend == MatMulBackend::Mlx),
        "expected MLX backend but observed {:?}",
        mlx_samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let flags = matmul_alpha_beta::take_last_mlx_alpha_beta_flags().expect("expected MLX alpha/beta flags to be recorded");
    assert_eq!(flags, (false, true), "expected write-only destination in scale-only mode");

    assert_eq!(expected.len(), actual.len());
    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - act).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            act,
            diff,
            rel
        );
    }

    let padded_view_after = value_history_view.as_mps_matrix_batch_view()?;
    assert!(
        padded_view_after.matrix_bytes > padded_view_after.rows * padded_view_after.row_bytes,
        "expected MLX path to consume padded batch without compacting"
    );

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_skinny_tile_single_row() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 1usize;
    let k = 64usize;
    let n = 128usize;

    let left_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.025 + 0.2).collect();
    let right_data: Vec<f32> = (0..(k * n)).map(|i| 1.0 - (i as f32) * 0.013).collect();
    let alpha = 0.75f32;
    let beta = 0.0f32;

    let left_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&left_data))?;
    let right_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&right_data))?;
    let result_data = vec![0.0f32; m * n];

    let expected = cpu_matmul_scaled(&left_data, m, k, &right_data, k, n, alpha, beta, None, false, false);

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor = make_result_tensor(&context, &result_data, &[m, n])?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();

    let samples = context.take_matmul_samples();
    assert!(
        samples.iter().any(|sample| matches!(sample.backend, MatMulBackend::Mlx)),
        "expected MLX backend samples but observed {:?}",
        samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let keys = context.kernel_manager.mlx_pipeline_keys();
    assert!(
        keys.iter().any(|key| {
            key.tile_shape == MlxTileShape::Tile1x32
                && !key.transpose_left
                && !key.transpose_right
                && !key.use_out_source
                && !key.do_axpby
                && key.scale_only
                && !key.has_batch
        }),
        "expected skinny MLX tile in pipeline cache, found {:?}",
        keys
    );

    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - got).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            got,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_skinny_tile_single_column() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 128usize;
    let k = 64usize;
    let n = 1usize;

    let left_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.019 + 0.35).collect();
    let right_data: Vec<f32> = (0..(k * n)).map(|i| 0.5 - (i as f32) * 0.009).collect();
    let alpha = 0.5f32;
    let beta = 0.0f32;

    let left_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&left_data))?;
    let right_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&right_data))?;
    let result_data = vec![0.0f32; m * n];

    let expected = cpu_matmul_scaled(&left_data, m, k, &right_data, k, n, alpha, beta, None, false, false);

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor = make_result_tensor(&context, &result_data, &[m, n])?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();

    let samples = context.take_matmul_samples();
    assert!(
        samples.iter().any(|sample| matches!(sample.backend, MatMulBackend::Mlx)),
        "expected MLX backend samples but observed {:?}",
        samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let keys = context.kernel_manager.mlx_pipeline_keys();
    assert!(
        keys.iter().any(|key| {
            key.tile_shape == MlxTileShape::Tile1x32
                && !key.transpose_left
                && !key.transpose_right
                && !key.use_out_source
                && !key.do_axpby
                && key.scale_only
                && !key.has_batch
        }),
        "expected skinny MLX tile in pipeline cache, found {:?}",
        keys
    );

    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - got).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            got,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_tile4_four_rows() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 4usize;
    let k = 64usize;
    let n = 128usize;
    let alpha = 0.75f32;
    let beta = 0.0f32;

    let left_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.021 - 0.2).collect();
    let right_data: Vec<f32> = (0..(k * n)).map(|i| 0.55 - (i as f32) * 0.013).collect();

    let left_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&left_data))?;
    let right_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&right_data))?;
    let result_data = vec![0.0f32; m * n];

    let expected = cpu_matmul_scaled(&left_data, m, k, &right_data, k, n, alpha, beta, None, false, false);

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor = make_result_tensor(&context, &result_data, &[m, n])?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &right_tensor, &result_tensor, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();

    let samples = context.take_matmul_samples();
    assert!(
        samples.iter().any(|sample| matches!(sample.backend, MatMulBackend::Mlx)),
        "expected MLX backend samples but observed {:?}",
        samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let keys = context.kernel_manager.mlx_pipeline_keys();
    assert!(
        keys.iter().any(|key| {
            key.tile_shape == MlxTileShape::Tile4x32
                && !key.transpose_left
                && !key.transpose_right
                && !key.use_out_source
                && !key.do_axpby
                && key.scale_only
                && !key.has_batch
        }),
        "expected Tile4x32 MLX tile in pipeline cache, found {:?}",
        keys
    );

    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - got).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            got,
            diff,
            rel
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_strided_kv_skinny_tile() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let batch_heads = 1usize;
    let cache_capacity = 8usize;
    let active_steps = 6usize;
    let query_len = 1usize;
    let head_dim = 128usize;

    let left_data: Vec<f32> = (0..(batch_heads * query_len * active_steps))
        .map(|i| (i as f32) * 0.0275 - 0.35)
        .collect();
    let cache_data: Vec<f32> = (0..(batch_heads * cache_capacity * head_dim))
        .map(|i| 0.65 - (i as f32) * 0.011)
        .collect();

    let left_tensor = Tensor::new(
        vec![batch_heads, query_len, active_steps],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&left_data),
    )?;
    let cache_tensor = Tensor::new(
        vec![batch_heads, cache_capacity, head_dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&cache_data),
    )?;

    let (history_view, _) = context.kv_cache_history_view(&cache_tensor, active_steps)?;
    let result_dims = [batch_heads, query_len, head_dim];
    let result_data = vec![0.0f32; result_dims.iter().product()];
    let alpha = 0.5f32;
    let beta = 0.0f32;

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMps);
    let result_tensor_mps = make_result_tensor(&context, &result_data, &result_dims)?;
    let expected_tensor = context.matmul_alpha_beta(&left_tensor, &history_view, &result_tensor_mps, false, false, alpha, beta)?;
    context.synchronize();
    let expected = expected_tensor.to_vec();
    context.take_matmul_samples();

    context.set_matmul_backend_preference(MatMulBackendPreference::ForceMlx);
    let result_tensor_mlx = make_result_tensor(&context, &result_data, &result_dims)?;
    let actual_tensor = context.matmul_alpha_beta(&left_tensor, &history_view, &result_tensor_mlx, false, false, alpha, beta)?;
    context.synchronize();
    let actual = actual_tensor.to_vec();

    let samples = context.take_matmul_samples();
    let non_total_backends: Vec<MatMulBackend> = samples
        .iter()
        .map(|sample| sample.backend)
        .filter(|backend| *backend != MatMulBackend::Total)
        .collect();
    assert!(
        !non_total_backends.is_empty() && non_total_backends.iter().all(|backend| *backend == MatMulBackend::Mlx),
        "expected MLX backend but observed {:?}",
        samples.iter().map(|sample| sample.backend).collect::<Vec<_>>()
    );

    let keys = context.kernel_manager.mlx_pipeline_keys();
    assert!(
        keys.iter().any(|key| {
            key.tile_shape == MlxTileShape::Tile1x32
                && !key.transpose_left
                && !key.transpose_right
                && !key.use_out_source
                && !key.do_axpby
                && key.scale_only
        }),
        "expected skinny MLX tile in pipeline cache, found {:?}",
        keys
    );

    let rtol = 1e-4f32;
    let atol = 1e-6f32;
    for (idx, (&exp, &got)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (exp - got).abs();
        let rel = if exp.abs() > 1e-6 { diff / exp.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at index {} (expected {}, got {}, diff {}, rel {})",
            idx,
            exp,
            got,
            diff,
            rel
        );
    }

    Ok(())
}
