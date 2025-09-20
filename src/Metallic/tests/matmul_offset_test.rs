use super::*;

#[test]
fn test_matmul_non_zero_buffer_offsets() -> Result<(), MetalError> {
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
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

    // Create buffers with the full data
    let a_buffer = create_buffer_with_data(&a_data_full, &context.device)?;
    let b_buffer = create_buffer_with_data(&b_data_full, &context.device)?;
    let c_buffer = create_buffer_with_data(&c_data_full, &context.device)?;

    // Create tensor views at offsets
    let a_tensor = Tensor::from_existing_buffer(
        a_buffer,
        vec![m, k],
        &context.device.clone(),
        offset_a * std::mem::size_of::<f32>(),
    )?;
    let b_tensor = Tensor::from_existing_buffer(
        b_buffer,
        vec![k, n],
        &context.device.clone(),
        offset_b * std::mem::size_of::<f32>(),
    )?;
    let c_tensor = Tensor::from_existing_buffer(
        c_buffer,
        vec![m, n],
        &context.device.clone(),
        offset_c * std::mem::size_of::<f32>(),
    )?;

    // Expected result: A * B
    // A = [[1, 1.1, 1.2], [1.3, 1.4, 1.5]] (from a_data_full[10..16])
    // B = [[3, 3.2, 3.4, 3.6], [3.8, 4.0, 4.2, 4.4]] (from b_data_full[15..21] reshaped)
    // But we actually have:
    // A = [[1, 1.1, 1.2], [1.3, 1.4, 1.5]]
    // B = [[3, 3.2], [3.4, 3.6], [3.8, 4.0]]
    // A * B = [[1*3+1.1*3.4+1.2*3.8, 1*3.2+1.1*3.6+1.2*4.0], [1.3*3+1.4*3.4+1.5*3.8, 1.3*3.2+1.4*3.6+1.5*4.0]]
    //       = [[3+3.74+4.56, 3.2+3.96+4.8], [3.9+4.76+5.7, 4.16+5.04+6.0]]
    //       = [[11.3, 11.96], [14.36, 15.2]]
    let expected_result = [11.3, 11.96, 14.36, 15.2];

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha: 1.0,
        beta: 0.0,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();
    let row_bytes_a = k * bytes_per_elem;
    let row_bytes_b = n * bytes_per_elem;
    let row_bytes_c = n * bytes_per_elem;

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: row_bytes_a,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: row_bytes_b,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: row_bytes_c,
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
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
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
    let _a_data: Vec<f32> = a_data_full[offset_a..offset_a + (m * k)].to_vec();
    let _b_data: Vec<f32> = b_data_full[offset_b..offset_b + (k * n)].to_vec();
    let _c_data: Vec<f32> = c_data_full[offset_c..offset_c + (m * n)].to_vec();

    // Create buffers with the full data
    let a_buffer = create_buffer_with_data(&a_data_full, &context.device)?;
    let b_buffer = create_buffer_with_data(&b_data_full, &context.device)?;
    let c_buffer = create_buffer_with_data(&c_data_full, &context.device)?;

    // Create tensor views at offsets
    let a_tensor = Tensor::from_existing_buffer(
        a_buffer,
        vec![m, k],
        &context.device,
        offset_a * std::mem::size_of::<f32>(),
    )?;
    let b_tensor = Tensor::from_existing_buffer(
        b_buffer,
        vec![k, n],
        &context.device,
        offset_b * std::mem::size_of::<f32>(),
    )?;
    let c_tensor = Tensor::from_existing_buffer(
        c_buffer,
        vec![m, n],
        &context.device,
        offset_c * std::mem::size_of::<f32>(),
    )?;

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
    let row_bytes_a = k * bytes_per_elem;
    let row_bytes_b = n * bytes_per_elem;
    let row_bytes_c = n * bytes_per_elem;

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: row_bytes_a,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: row_bytes_b,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: row_bytes_c,
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
    let context = Context::new()?;
    let mut cache = ResourceCache::new();
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

    // Create buffers with the full data
    let a_buffer = create_buffer_with_data(&a_data_full, &context.device)?;
    let b_buffer = create_buffer_with_data(&b_data_full, &context.device)?;
    let c_buffer = create_buffer_with_data(&c_data_full, &context.device)?;

    // Create tensor views at offsets
    let a_tensor = Tensor::from_existing_buffer(
        a_buffer,
        vec![m, k],
        &context.device.clone(),
        offset_a * std::mem::size_of::<f32>(),
    )?;
    let b_tensor = Tensor::from_existing_buffer(
        b_buffer,
        vec![k, n],
        &context.device.clone(),
        offset_b * std::mem::size_of::<f32>(),
    )?;
    let c_tensor = Tensor::from_existing_buffer(
        c_buffer,
        vec![m, n],
        &context.device.clone(),
        offset_c * std::mem::size_of::<f32>(),
    )?;

    // Expected result: A * B (validated with NumPy)
    // A = [[5.0, 5.01], [5.02, 5.03], [5.04, 5.05]]
    // B = [[12.0, 12.02, 12.04, 12.06], [12.08, 12.10, 12.12, 12.14]]
    let expected_result = [
        120.5208, 120.721, 120.9212, 121.1214, 121.0024, 121.2034, 121.4044, 121.6054, 121.484,
        121.6858, 121.8876, 122.0894,
    ];

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha: 1.0,
        beta: 0.0,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key.clone(), &context.device)?;

    let bytes_per_elem: usize = core::mem::size_of::<f32>();
    let row_bytes_a = k * bytes_per_elem;
    let row_bytes_b = n * bytes_per_elem;
    let row_bytes_c = n * bytes_per_elem;

    let a_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: k,
        row_bytes: row_bytes_a,
    };
    let b_desc_key = MpsMatrixDescriptorKey {
        rows: k,
        columns: n,
        row_bytes: row_bytes_b,
    };
    let result_desc_key = MpsMatrixDescriptorKey {
        rows: m,
        columns: n,
        row_bytes: row_bytes_c,
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

// Helper function to create a Metal buffer with data
fn create_buffer_with_data(
    data: &[f32],
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_len = std::mem::size_of_val(data);
    let item_ptr =
        std::ptr::NonNull::new(data.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

    let buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                item_ptr,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(MetalError::BufferFromBytesCreationFailed)?
    };

    Ok(buf)
}
