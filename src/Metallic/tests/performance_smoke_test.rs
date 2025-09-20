use super::*;

#[test]
fn test_performance_smoke_small() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 1;
    let seq_q = 32;
    let seq_k = 32;
    let dim = 64;

    let q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    // Warm up
    let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;

    // Measure performance
    let start = Instant::now();
    let iterations = 10;

    for _ in 0..iterations {
        let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;
    }

    let duration = start.elapsed();
    let avg_time = duration.as_millis() as f64 / iterations as f64;

    // This is just a smoke test - we're not asserting on specific performance,
    // just that it completes in a reasonable time (less than 100ms per iteration)
    assert!(
        avg_time < 100.0,
        "Performance regression detected: average time {:.2}ms > 100ms",
        avg_time
    );

    Ok(())
}

#[test]
fn test_performance_smoke_medium() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 2;
    let seq_q = 128;
    let seq_k = 128;
    let dim = 128;

    let q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    // Warm up
    let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;

    // Measure performance
    let start = Instant::now();
    let iterations = 5;

    for _ in 0..iterations {
        let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?;
    }

    let duration = start.elapsed();
    let avg_time = duration.as_millis() as f64 / iterations as f64;

    // This is just a smoke test - we're not asserting on specific performance,
    // just that it completes in a reasonable time (less than 500ms per iteration)
    assert!(
        avg_time < 500.0,
        "Performance regression detected: average time {:.2}ms > 500ms",
        avg_time
    );

    Ok(())
}

#[test]
fn test_performance_smoke_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let batch = 1;
    let seq_q = 64;
    let seq_k = 64;
    let dim = 64;

    let q_data: Vec<f32> = (0..(batch * seq_q * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim))
        .map(|i| (i as f32) * 0.001)
        .collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    // Warm up
    let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)?;

    // Measure performance
    let start = Instant::now();
    let iterations = 10;

    for _ in 0..iterations {
        let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)?;
    }

    let duration = start.elapsed();
    let avg_time = duration.as_millis() as f64 / iterations as f64;

    // This is just a smoke test - we're not asserting on specific performance,
    // just that it completes in a reasonable time (less than 200ms per iteration)
    assert!(
        avg_time < 200.0,
        "Performance regression detected: average time {:.2}ms > 200ms",
        avg_time
    );

    Ok(())
}

#[test]
fn test_performance_matmul_small() -> Result<(), MetalError> {
    let context = Context::new()?;
    let mut cache = super::super::resource_cache::ResourceCache::new();

    let m = 32;
    let k = 64;
    let n = 32;

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
    let result_data: Vec<f32> = vec![0.0; m * n];

    let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
    let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
    let result_tensor = Tensor::create_tensor_from_slice(&result_data, vec![m, n], &context)?;

    use super::super::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
    use super::super::matmul::MatMulOperation;

    let gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: m,
        result_columns: n,
        interior_columns: k,
        alpha: 1.0,
        beta: 0.0,
    };
    let gemm_op = cache.get_or_create_gemm(gemm_key, &context.device)?;

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
        result_buf: result_tensor.buf.clone(),
        result_offset: result_tensor.offset,
        left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
        right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
        result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
        gemm: gemm_op,
    };

    // Warm up
    let command_buffer = context.command_queue.commandBuffer().unwrap();
    matmul_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Measure performance
    let start = Instant::now();
    let iterations = 20;

    for _ in 0..iterations {
        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }
    }

    let duration = start.elapsed();
    let avg_time = duration.as_micros() as f64 / iterations as f64;

    // This is just a smoke test - we're not asserting on specific performance,
    // just that it completes in a reasonable time (less than 1500 microseconds per iteration)
    assert!(
        avg_time < 1500.0,
        "Performance regression detected: average time {:.2}μs > 1500μs",
        avg_time
    );

    Ok(())
}

#[test]
fn test_performance_softmax_small() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    super::super::ensure_fused_softmax_pipeline(&mut context)?;

    let seq_q = 32;
    let seq_k = 32;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.01).collect();
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims, &context)?;

    use super::super::resource_cache::ResourceCache;
    use super::super::softmax::SoftmaxOperation;

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    // Warm up
    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    // Measure performance with multiple trials to account for system noise
    let trials = 5;
    let iterations_per_trial = 10;
    let mut trial_times = Vec::with_capacity(trials);

    for _ in 0..trials {
        let start = Instant::now();

        for _ in 0..iterations_per_trial {
            let command_buffer = context.command_queue.commandBuffer().unwrap();
            sm_op.encode(&command_buffer, &mut cache)?;
            command_buffer.commit();
            unsafe {
                command_buffer.waitUntilCompleted();
            }
        }

        let duration = start.elapsed();
        let avg_time = duration.as_micros() as f64 / iterations_per_trial as f64;
        trial_times.push(avg_time);
    }

    // Take the minimum time as our performance metric to reduce noise from system interference
    let min_time = trial_times.into_iter().fold(f64::INFINITY, f64::min);

    // This is just a smoke test - we're not asserting on specific performance,
    // just that it completes in a reasonable time (less than 500 microseconds per iteration)
    assert!(
        min_time < 500.0,
        "Performance regression detected: minimum time {:.2}μs > 500μs",
        min_time
    );

    Ok(())
}
