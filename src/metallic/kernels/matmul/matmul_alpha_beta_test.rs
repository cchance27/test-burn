use super::*;
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
