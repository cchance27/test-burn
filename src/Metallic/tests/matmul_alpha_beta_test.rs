
use super::*;

#[test]
fn test_matmul_alpha_beta_accumulation() -> Result<(), MetalError> {
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
    let m = 2;
    let k = 3;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix
    let c_data = vec![0.5, 1.5, 2.5, 3.5]; // 2x2 matrix (will be used as result with beta)

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
    let c_tensor = Tensor::create_tensor_from_slice(&c_data, vec![m, n], &context)?; // Will be used as result with beta

    let alpha = 0.5;
    let beta = 0.25;

    // Expected result: alpha * A * B + beta * C
    // A * B = [[31, 19], [85, 55]] (calculated manually)
    // alpha * A * B = [[15.5, 9.5], [42.5, 27.5]]
    // beta * C = [[0.125, 0.375], [0.625, 0.875]]
    // Final result = [[15.625, 9.875], [43.125, 28.375]]
    let expected_result = vec![15.625, 9.875, 43.125, 28.375];

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha,
        beta,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: k * bytes_per_elem,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };

    let matmul_op = MatMulOperation {
        left_buf: a_tensor.buf.clone(),
        left_offset: a_tensor.offset,
        right_buf: b_tensor.buf.clone(),
        right_offset: b_tensor.offset,
        result_buf: c_tensor.buf.clone(), // Use C as the result buffer for beta accumulation
        result_offset: c_tensor.offset,
        left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
        right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
        result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
        gemm: gemm_op,
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    matmul_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = c_tensor.as_slice();

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
fn test_matmul_alpha_beta_with_different_values() -> Result<(), MetalError> {
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
    let m = 3;
    let k = 2;
    let n = 4;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.5).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.3).collect();
    let c_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.1).collect();

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
    let c_tensor = Tensor::create_tensor_from_slice(&c_data, vec![m, n], &context)?;

    let alpha = 2.0;
    let beta = -0.5;

    // Calculate expected result manually
    // A = [[0, 0.5], [1, 1.5], [2, 2.5]]
    // B = [[0, 0.3, 0.6, 0.9], [1.2, 1.5, 1.8, 2.1]]
    // A * B = [[0.6, 0.75, 0.9, 1.05], [1.8, 2.55, 3.3, 4.05], [3.0, 4.35, 5.7, 7.05]]
    // alpha * A * B = [[1.2, 1.5, 1.8, 2.1], [3.6, 5.1, 6.6, 8.1], [6.0, 8.7, 11.4, 14.1]]
    // beta * C = [[0, -0.05, -0.1, -0.15], [-0.2, -0.25, -0.3, -0.35], [-0.4, -0.45, -0.5, -0.55]]
    // Final result = [[1.2, 1.45, 1.7, 1.95], [3.4, 4.85, 6.3, 7.75], [5.6, 8.25, 10.9, 13.55]]
    let expected_result = vec![
        1.2, 1.45, 1.7, 1.95, 3.4, 4.85, 6.3, 7.75, 5.6, 8.25, 10.9, 13.55,
    ];

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha,
        beta,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: k * bytes_per_elem,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };

    let matmul_op = MatMulOperation {
        left_buf: a_tensor.buf.clone(),
        left_offset: a_tensor.offset,
        right_buf: b_tensor.buf.clone(),
        right_offset: b_tensor.offset,
        result_buf: c_tensor.buf.clone(),
        result_offset: c_tensor.offset,
        left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
        right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
        result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
        gemm: gemm_op,
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    matmul_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = c_tensor.as_slice();

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
fn test_matmul_alpha_zero_beta_one() -> Result<(), MetalError> {
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let c_data = vec![10.0, 20.0, 30.0, 40.0]; // This should be the result when alpha=0, beta=1

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
    let c_tensor = Tensor::create_tensor_from_slice(&c_data, vec![m, n], &context)?;

    let alpha = 0.0;
    let beta = 1.0;

    // Expected result: 0 * A * B + 1 * C = C
    let expected_result = c_data.clone();

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha,
        beta,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: k * bytes_per_elem,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };

    let matmul_op = MatMulOperation {
        left_buf: a_tensor.buf.clone(),
        left_offset: a_tensor.offset,
        right_buf: b_tensor.buf.clone(),
        right_offset: b_tensor.offset,
        result_buf: c_tensor.buf.clone(),
        result_offset: c_tensor.offset,
        left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
        right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
        result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
        gemm: gemm_op,
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    matmul_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = c_tensor.as_slice();

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
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
    let m = 2;
    let k = 2;
    let n = 2;

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let c_data = vec![0.0, 0.0, 0.0, 0.0]; // This will be overwritten when alpha=1, beta=0

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
    let c_tensor = Tensor::create_tensor_from_slice(&c_data, vec![m, n], &context)?;

    let alpha = 1.0;
    let beta = 0.0;

    // Expected result: 1 * A * B + 0 * C = A * B
    // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected_result = vec![19.0, 22.0, 43.0, 50.0];

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha,
        beta,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: k * bytes_per_elem,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: n * bytes_per_elem,
    };

    let matmul_op = MatMulOperation {
        left_buf: a_tensor.buf.clone(),
        left_offset: a_tensor.offset,
        right_buf: b_tensor.buf.clone(),
        right_offset: b_tensor.offset,
        result_buf: c_tensor.buf.clone(),
        result_offset: c_tensor.offset,
        left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
        right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
        result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
        gemm: gemm_op,
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    matmul_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = c_tensor.as_slice();

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
