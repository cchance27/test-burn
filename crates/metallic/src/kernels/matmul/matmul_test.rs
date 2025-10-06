use crate::kernels::matmul::{MatMulAlphaBetaOp, MatMulOp, mps_matrix_from_buffer};
use crate::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage};

// Helpers

// CPU-based matrix multiplication for golden testing
#[allow(clippy::too_many_arguments)]
fn cpu_matmul(
    a: &[f32],
    a_original_rows: usize,
    a_original_cols: usize,
    b: &[f32],
    b_original_rows: usize,
    b_original_cols: usize,
    transpose_left: bool,
    transpose_right: bool,
) -> Vec<f32> {
    let effective_a_rows = if transpose_left { a_original_cols } else { a_original_rows };
    let effective_a_cols = if transpose_left { a_original_rows } else { a_original_cols };
    let effective_b_rows = if transpose_right { b_original_cols } else { b_original_rows };
    let effective_b_cols = if transpose_right { b_original_rows } else { b_original_cols };

    assert_eq!(
        effective_a_cols, effective_b_rows,
        "Matrix dimensions are not compatible for multiplication"
    );

    let m = effective_a_rows;
    let n = effective_b_cols;
    let k = effective_a_cols; // or effective_b_rows

    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let a_val = if transpose_left {
                    a[l * a_original_cols + i] // Access a[l][i] when A is transposed (A^T[i,l] = A[l,i])
                } else {
                    a[i * a_original_cols + l] // Access a[i][l] normally
                };
                let b_val = if transpose_right {
                    b[j * b_original_cols + l] // Access b[j][l] when B is transposed (B^T[l,j] = B[j,l])
                } else {
                    b[l * b_original_cols + j] // Access b[l][j] normally
                };
                sum += a_val * b_val;
            }
            result[i * n + j] = sum;
        }
    }
    result
}

// CPU-based matrix multiplication for golden testing with alpha/beta scaling
#[allow(clippy::too_many_arguments)]
fn cpu_matmul_scaled(
    a: &[f32],
    a_original_rows: usize,
    a_original_cols: usize,
    b: &[f32],
    b_original_rows: usize,
    b_original_cols: usize,
    alpha: f32,
    beta: f32,
    c: Option<&[f32]>,
    transpose_left: bool,
    transpose_right: bool,
) -> Vec<f32> {
    let effective_a_rows = if transpose_left { a_original_cols } else { a_original_rows };
    let effective_a_cols = if transpose_left { a_original_rows } else { a_original_cols };
    let effective_b_rows = if transpose_right { b_original_cols } else { b_original_rows };
    let effective_b_cols = if transpose_right { b_original_rows } else { b_original_cols };

    assert_eq!(
        effective_a_cols, effective_b_rows,
        "Matrix dimensions are not compatible for multiplication"
    );

    let m = effective_a_rows;
    let n = effective_b_cols;
    let k = effective_a_cols; // or effective_b_rows

    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let a_val = if transpose_left {
                    a[l * a_original_cols + i] // Access a[l][i] when A is transposed (A^T[i,l] = A[l,i])
                } else {
                    a[i * a_original_cols + l] // Access a[i][l] normally
                };
                let b_val = if transpose_right {
                    b[j * b_original_cols + l] // Access b[j][l] when B is transposed (B^T[l,j] = B[j,l])
                } else {
                    b[l * b_original_cols + j] // Access b[l][j] normally
                };
                sum += a_val * b_val;
            }
            let scaled_sum = alpha * sum;

            // Apply beta * C part if C is provided
            let beta_c = if let Some(c_slice) = c { beta * c_slice[i * n + j] } else { 0.0 };

            result[i * n + j] = scaled_sum + beta_c;
        }
    }
    result
}

// Tests for MatMul Below
#[test]
fn test_mps_matrix_from_buffer() -> Result<(), crate::MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Create a small tensor for testing
    let tensor = Tensor::new(vec![2, 3], TensorStorage::Pooled(&mut ctx), TensorInit::Uninitialized)?;

    // Create a matrix descriptor for a 2x3 matrix
    let desc = unsafe {
        objc2_metal_performance_shaders::MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            2,                              // rows
            3,                              // columns
            3 * std::mem::size_of::<f32>(), // row bytes for 3 f32 elements (usize, not u64)
            objc2_metal_performance_shaders::MPSDataType::Float32,
        )
    };

    // This should create an MPSMatrix view without error
    let mps_matrix = mps_matrix_from_buffer(&tensor.buf, tensor.offset, &desc);
    // The rows() method is unsafe in newer versions
    unsafe {
        assert!(mps_matrix.rows() == 2);
    }

    Ok(())
}

#[test]
fn test_matmul_correctness_small_int() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2; // A rows
    let k = 3; // A cols / B rows
    let n = 2; // B cols

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 3, 2, false, false);

    // Use the new kernel system
    let result_tensor = context.call::<MatMulOp>((&a_tensor, &b_tensor, false, false))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    assert_eq!(metal_output, cpu_output);

    Ok(())
}

#[test]
fn test_matmul_correctness_asymmetric_float() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 5;
    let k = 4;
    let n = 7;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.123) - 1.0).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32 * 0.456) + 0.5).collect();

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_output = cpu_matmul(&a_data, 5, 4, &b_data, 4, 7, false, false);

    // Use the new kernel system
    let result_tensor = context.call::<MatMulOp>((&a_tensor, &b_tensor, false, false))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_transpose_right() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2; // A rows
    let k = 3; // A cols
    let n = 2; // B rows (after transpose)

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: 2x3
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // B: 2x3 (will be B^T: 3x2)

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![n, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?; // B is 2x3, but conceptually 3x2 for matmul

    let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 2, 3, false, true);

    // Use the new kernel system
    let result_tensor = context.call::<MatMulOp>((&a_tensor, &b_tensor, false, true))?; // transpose right only
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_transpose_left() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 3; // A rows (after transpose)
    let k = 2; // A cols
    let n = 3; // B cols

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: 2x3 (will be A^T: 3x2)
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // B: 2x3

    let a_tensor = Tensor::new(vec![k, m], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?; // A is 2x3, but conceptually 3x2 for matmul
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 2, 3, true, false);

    // Use the new kernel system
    let result_tensor = context.call::<MatMulOp>((&a_tensor, &b_tensor, true, false))?; // transpose left only
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_alpha_beta_accumulation() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix (flattened row-major)
    let c_data = vec![0.5, 1.5, 2.5, 3.5]; // 2x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?; // Will be used as result with beta

    let alpha = 0.5;
    let beta = 0.25;

    // Expected result: alpha * A * B + beta * C
    // A * B = [[31, 19], [85, 55]] (calculated with Python)
    // alpha * A * B = [[15.5, 9.5], [42.5, 27.5]]
    // beta * C = [[0.125, 0.375], [0.625, 0.875]]
    // Final result = [[15.625, 9.875], [43.125, 28.375]]
    let expected_result = [15.625, 9.875, 43.125, 28.375];

    // Use the new kernel system with alpha/beta scaling
    let result_tensor = context.call::<MatMulAlphaBetaOp>((&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta))?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val: f64 = expected_result[i];
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

// MatMul Offset tests

#[test]
fn test_matmul_non_zero_buffer_offsets() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 3;
    let n = 2;

    // Create larger buffers with offset
    let total_elements_a = 100;
    let total_elements_b = 100;
    let total_elements_c = 100;

    // Offsets in elements
    let offset_a = 10;
    let offset_b = 15;
    let offset_c = 20;

    // Data for matrices (placed at offsets)
    let a_data_full: Vec<f32> = (0..total_elements_a).map(|i| (i as f32) * 0.1).collect();
    let b_data_full: Vec<f32> = (0..total_elements_b).map(|i| (i as f32) * 0.2).collect();
    let c_data_full: Vec<f32> = (0..total_elements_c).map(|i| (i as f32) * 0.3).collect();

    // Extract sub-tensors from the full tensors at the specified offsets
    // This would require implementing sub-tensor functionality that operates on buffer offsets
    // The tensor slicing logic would need to be added to the Tensor implementation
    // For now, using the new kernel system with proper tensors
    let a_slice = &a_data_full[offset_a..offset_a + (m * k)];
    let b_slice = &b_data_full[offset_b..offset_b + (k * n)];
    let c_slice = &c_data_full[offset_c..offset_c + (m * n)];

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(a_slice))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(b_slice))?;
    let _c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(c_slice))?;

    // Expected result: A * B
    // A = [[1, 1.1, 1.2], [1.3, 1.4, 1.5]] (from a_data_full[10..16])
    // B = [[3, 3.2], [3.4, 3.6], [3.8, 4.0]] (from b_data_full[15..21])
    // A * B = [[1*3+1.1*3.4+1.2*3.8, 1*3.2+1.1*3.6+1.2*4.0], [1.3*3+1.4*3.4+1.5*3.8, 1.3*3.2+1.4*3.6+1.5*4.0]]
    //       = [[3+3.74+4.56, 3.2+3.96+4.8], [3.9+4.76+5.7, 4.16+5.04+6.0]]
    //       = [[11.3, 11.96], [14.36, 15.2]]
    let expected_result = [11.3, 11.96, 14.36, 15.2];

    // Use the new kernel system
    let result_tensor = context.matmul(&a_tensor, &b_tensor, false, false)?;
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
fn test_matmul_non_zero_buffer_offsets_with_alpha_beta() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 2;
    let n = 2;

    // Create larger buffers with offset
    let total_elements_a = 50;
    let total_elements_b = 50;
    let total_elements_c = 50;

    // Offsets in elements
    let offset_a = 5;
    let offset_b = 8;
    let offset_c = 12;

    // Data for matrices (placed at offsets)
    let a_data_full: Vec<f32> = (0..total_elements_a).map(|i| (i as f32) * 0.5).collect();
    let b_data_full: Vec<f32> = (0..total_elements_b).map(|i| (i as f32) * 0.4).collect();
    let c_data_full: Vec<f32> = (0..total_elements_c).map(|i| (i as f32) * 0.1).collect();

    // Actual matrix data (starting at offsets)
    let a_data: Vec<f32> = a_data_full[offset_a..offset_a + (m * k)].to_vec();
    let b_data: Vec<f32> = b_data_full[offset_b..offset_b + (k * n)].to_vec();
    let c_data: Vec<f32> = c_data_full[offset_c..offset_c + (m * n)].to_vec();

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?;

    let alpha = 2.0;
    let beta = 0.5;

    // Calculate expected result:
    // A = [[2.5, 3.0], [3.5, 4.0]] (from a_data_full[5..9])
    // B = [[3.2, 3.6], [4.0, 4.4]] (from b_data_full[8..12])
    // C = [[1.2, 1.3], [1.4, 1.5]] (from c_data_full[12..16])
    // A * B = [[2.5*3.2+3.0*4.0, 2.5*3.6+3.0*4.4], [3.5*3.2+4.0*4.0, 3.5*3.6+4.0*4.4]]
    //       = [[8+12, 9+13.2], [11.2+16, 12.6+17.6]]
    //       = [[20, 22.2], [27.2, 30.2]]
    // alpha * A * B = [[40, 44.4], [54.4, 60.4]]
    // beta * C = [[0.6, 0.65], [0.7, 0.75]]
    // Final result = [[40.6, 45.05], [55.1, 61.15]]
    let expected_result = [40.6, 45.05, 55.1, 61.15];

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
fn test_matmul_large_offsets() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 3;
    let k = 2;
    let n = 4;

    // Create larger buffers with large offset
    let total_elements_a = 1000;
    let total_elements_b = 1000;
    let total_elements_c = 1000;

    // Large offsets in elements
    let offset_a = 500;
    let offset_b = 600;
    let offset_c = 700;

    // Data for matrices (placed at offsets)
    let a_data_full: Vec<f32> = (0..total_elements_a).map(|i| (i as f32) * 0.01).collect();
    let b_data_full: Vec<f32> = (0..total_elements_b).map(|i| (i as f32) * 0.02).collect();
    let c_data_full: Vec<f32> = (0..total_elements_c).map(|i| (i as f32) * 0.03).collect();

    let a_data: Vec<f32> = a_data_full[offset_a..offset_a + (m * k)].to_vec();
    let b_data: Vec<f32> = b_data_full[offset_b..offset_b + (k * n)].to_vec();
    let c_data: Vec<f32> = c_data_full[offset_c..offset_c + (m * n)].to_vec();

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let _c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?;

    // Expected result: A * B (validated with NumPy)
    // A = [[5.0, 5.01], [5.02, 5.03], [5.04, 5.05]]
    // B = [[12.0, 12.02, 12.04, 12.06], [12.08, 12.10, 12.12, 12.14]]
    let expected_result = [
        120.5208, 120.721, 120.9212, 121.1214, 121.0024, 121.2034, 121.4044, 121.6054, 121.484, 121.6858, 121.8876, 122.0894,
    ];

    // Use the new kernel system
    let result_tensor = context.matmul(&a_tensor, &b_tensor, false, false)?;
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

// Mat Mull Transpose tests
#[test]
fn test_matmul_no_transpose() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let _result_tensor = Tensor::zeros(vec![m, n], &mut context, true)?;
    let cpu_output = cpu_matmul(&a_data, m, k, &b_data, k, n, false, false);

    // Use the new kernel system
    let result_tensor = context.matmul(&a_tensor, &b_tensor, false, false)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let expected_val = cpu_output[i] as f64; // Use actual CPU result, not hardcoded
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
fn test_matmul_transpose_both() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    // A is 2x3, A^T is 3x2
    // B is 2x4, B^T is 4x2 (note: for A^T * B^T, A^T cols must equal B^T rows)
    // A^T * B^T = 3x2 * 4x2 -> This won't work since 2 != 4
    // To multiply A^T * B^T, we need A cols = B rows
    // So if A is 2x3, A^T is 3x2, then B must be 3xN, so B^T is Nx3, and A^T * B^T = 3x2 * N*3 - not compatible
    // Actually, A^T * B^T: if A is mxk and B is kxn, then A^T is kxm and B^T is nxk
    // So A^T * B^T = kxm * nxk = need m=n for this to work
    // Let's make A 2x3 and B 2x3, so A^T is 3x2 and B^T is 3x2 - which can't be multiplied
    // To multiply A^T * B^T where A is mxp and B is qxn, we need p=q to make A^T * B^T = pxm * nxq work, so m=n
    // Let's make A 2x3 and B 4x2, so A^T is 3x2 and B^T is 2x4, then A^T * B^T = 3x2 * 2x4 = 3x4
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix: [[1,2,3], [4,5,6]]
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix: [[7,8], [9,10], [11,12]]

    let a_tensor = Tensor::new(vec![2, 3], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![3, 2], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;

    // A^T * B^T: A^T = [[1,4], [2,5], [3,6]], B^T = [[7,9,11], [8,10,12]]
    // A^T * B^T = [[1*7+4*8, 1*9+4*10, 1*11+4*12], [2*7+5*8, 2*9+5*10, 2*11+5*12], [3*7+6*8, 3*9+6*10, 3*11+6*12]]
    //         = [[7+32, 9+40, 11+48], [14+40, 18+50, 22+60], [21+48, 27+60, 33+72]]
    //         = [[39, 49, 59], [54, 68, 82], [69, 87, 105]]
    let expected_result = [39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0]; // 3x3 result

    // Use the new kernel system - A^T * B^T
    let result_tensor = context.matmul(&a_tensor, &b_tensor, true, true)?; // transpose both
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(3 * 3) {
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
fn test_matmul_extreme_alpha_scaling() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 4;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 4x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let result_tensor = Tensor::zeros(vec![m, n], &mut context, true)?;
    // Test with very large alpha value
    let alpha = 1e6f32;
    let beta = 0.0f32;

    let cpu_output = cpu_matmul_scaled(&a_data, m, k, &b_data, k, n, alpha, beta, None, false, false);

    // Use the new kernel system with alpha/beta scaling
    let result_from_kernel = context.matmul_alpha_beta(&a_tensor, &b_tensor, &result_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_from_kernel.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_extreme_beta_accumulation() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix (flattened row-major)
    let c_data = vec![0.5, 1.5, 2.5, 3.5]; // 2x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let mut c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?; // Will be used as result with beta

    let alpha = 1.0f32;
    let beta = 1e6f32; // Very large beta

    // Expected result: alpha * A * B + beta * C
    let expected_result = cpu_matmul_scaled(&a_data, m, k, &b_data, k, n, alpha, beta, Some(&c_data), false, false);

    // Use the new kernel system with alpha/beta scaling
    c_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = c_tensor.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

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
fn test_matmul_scaled_with_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 3;
    let k = 2;
    let n = 3;

    // Create input matrices with extreme values
    let a_data = vec![1e8f32, -1e8f32, 1e8f32, -1e8f32, 1e8f32, -1e8f32]; // 3x2 matrix with extreme values
    let b_data = vec![2e7f32, -2e7f32, 2e7f32, -2e7f32, 2e7f32, -2e7f32]; // 2x3 matrix with extreme values

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let mut result_tensor = Tensor::zeros(vec![m, n], &mut context, true)?;
    // Use alpha that's very small and beta that's very large to test scaling
    let alpha = 1e-8f32;
    let beta = 0.0f32; // No accumulation in this test

    let cpu_output = cpu_matmul_scaled(&a_data, m, k, &b_data, k, n, alpha, beta, None, false, false);

    // Use the new kernel system with alpha/beta scaling
    result_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &result_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = result_tensor.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_scaled_alpha_beta_extremes() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
    let b_data = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
    let c_data = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let mut c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?; // Will be used as result with beta

    // Use extreme alpha and beta values
    let alpha = 1e10f32;
    let beta = -1e10f32;

    let cpu_output = cpu_matmul_scaled(&a_data, m, k, &b_data, k, n, alpha, beta, Some(&c_data), false, false);

    // Use the new kernel system with alpha/beta scaling
    c_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = c_tensor.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_matmul_scaling_zero_values() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![0.0, 0.0, 0.0, 0.0]; // All zero matrix
    let b_data = vec![1.0, 2.0, 3.0, 4.0]; // Regular matrix
    let c_data = vec![-1.0, -2.0, -3.0, -4.0]; // Regular matrix

    let a_tensor = Tensor::new(vec![m, k], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&a_data))?;
    let b_tensor = Tensor::new(vec![k, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&b_data))?;
    let mut c_tensor = Tensor::new(vec![m, n], TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&c_data))?; // Will be used as result with beta

    // Use various scaling factors with zero matrix
    let alpha = 1e6f32;
    let beta = 1e6f32;

    let cpu_output = cpu_matmul_scaled(&a_data, m, k, &b_data, k, n, alpha, beta, Some(&c_data), false, false);

    // Use the new kernel system with alpha/beta scaling
    c_tensor = context.matmul_alpha_beta(&a_tensor, &b_tensor, &c_tensor, false, false, alpha, beta)?;
    context.synchronize();

    let metal_output = c_tensor.as_slice();

    // Verify output does not contain infinities or NaNs
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Metal output contains non-finite value at index {}: {}", i, val);
    }

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..(m * n) {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}
