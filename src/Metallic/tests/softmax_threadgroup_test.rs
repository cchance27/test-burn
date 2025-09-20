use super::*;

#[test]
fn test_softmax_threadgroup_execution_width() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;

    // Get the pipeline
    let pipeline = context.fused_softmax_pipeline.as_ref().unwrap();

    // Check that the pipeline has a valid execution width
    let execution_width = pipeline.threadExecutionWidth();
    assert!(
        execution_width >= 32,
        "Execution width should be at least 32, got {}",
        execution_width
    );

    // Test with a small attention matrix that fits within a single threadgroup
    let seq_q = 4;
    let seq_k = 8;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: pipeline.clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Verify the operation completed successfully
    let metal_output = attn_tensor.as_slice();
    // Check that all values are finite
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value at index {}: {}", i, val);
    }

    // Verify row sums to ~1.0
    for i in 0..seq_q {
        let row_sum: f32 = metal_output[i * seq_k..(i + 1) * seq_k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );
    }

    Ok(())
}

#[test]
fn test_softmax_threadgroup_large_seq_k() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;

    // Get the pipeline
    let pipeline = context.fused_softmax_pipeline.as_ref().unwrap();

    // Test with a larger attention matrix that requires multiple threadgroups
    let seq_q = 2;
    let seq_k = 256; // This should require multiple threadgroups
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.01).collect();
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: pipeline.clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Verify the operation completed successfully
    let metal_output = attn_tensor.as_slice();
    // Check that all values are finite
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value at index {}: {}", i, val);
    }

    // Verify row sums to ~1.0
    for i in 0..seq_q {
        let row_sum: f32 = metal_output[i * seq_k..(i + 1) * seq_k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );
    }

    Ok(())
}

#[test]
fn test_softmax_threadgroup_very_large_seq_k() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;

    // Get the pipeline
    let pipeline = context.fused_softmax_pipeline.as_ref().unwrap();

    // Test with a very large attention matrix
    let seq_q = 1;
    let seq_k = 512; // This should require multiple threadgroups
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.005).collect();
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 1, // Test with causal masking as well
        pipeline: pipeline.clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Verify the operation completed successfully
    let metal_output = attn_tensor.as_slice();
    // Check that all values are finite
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value at index {}: {}", i, val);
    }

    // Verify row sums to ~1.0 and masked elements are zeroed
    let row_slice = &metal_output[0..seq_k];
    let row_sum: f32 = row_slice.iter().sum();
    assert!(
        (row_sum - 1.0).abs() < 1e-5,
        "Row sum mismatch: expected ~1.0, got {:.6}",
        row_sum
    );

    // Check masked elements (all except the first element should be zeroed for causal)
    (1..seq_k).for_each(|j| {
        assert!(
            row_slice[j].abs() < 1e-5,
            "Masked element at index {} not zeroed: got {:.6}",
            j,
            row_slice[j]
        );
    });

    Ok(())
}

#[test]
fn test_softmax_threadgroup_multiple_rows() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;

    // Get the pipeline
    let pipeline = context.fused_softmax_pipeline.as_ref().unwrap();

    // Test with multiple rows to ensure threadgroup handling works across rows
    let seq_q = 16; // Multiple rows
    let seq_k = 64; // Reasonable size per row
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.02).collect();
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: pipeline.clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Verify the operation completed successfully
    let metal_output = attn_tensor.as_slice();
    // Check that all values are finite
    for (i, &val) in metal_output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value at index {}: {}", i, val);
    }

    // Verify row sums to ~1.0 for all rows
    for i in 0..seq_q {
        let row_sum: f32 = metal_output[i * seq_k..(i + 1) * seq_k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );
    }

    Ok(())
}
