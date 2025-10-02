use super::*;

use super::matmul_alpha_beta;
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
