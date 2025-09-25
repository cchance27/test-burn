use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::scaled_dot_product_attention::scaled_dot_product_attention_impl;
use crate::metallic::*;

#[test]
fn test_sdpa_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Ensure the fused softmax pipeline is available
    ensure_fused_softmax_pipeline(&mut context)?;

    let seq_q = 4;
    let seq_k = 4;
    let dim = 4;
    let batch = 1;

    // Create tensor with very large values
    let large_value = 1e6f32;
    let q_data: Vec<f32> = (0..(batch * seq_q * dim)).map(|_| large_value).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| large_value).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| large_value).collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?; // causal = false

    // Synchronize to ensure the SDPA operation is complete before reading values
    context.synchronize();

    let output = result.as_slice();
    println!("Large values output: {:?}", output);

    // Verify output does not contain infinities or NaNs
    for &val in output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }

    Ok(())
}

#[test]
fn test_sdpa_extreme_negative_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;
    // Test with very negative values in query, key, and value tensors
    let batch = 1;
    let seq_q = 3;
    let seq_k = 3;
    let dim = 2;
    let out_tensor = Tensor::zeros(vec![batch, seq_q, dim], &mut context)?;
    let attn_tensor = Tensor::zeros(vec![batch, seq_q, seq_k], &mut context)?;
    let device = &context.device;
    let command_queue = &context.command_queue;

    // Create tensor with very negative values
    let negative_value = -1e6f32;
    let q_data: Vec<f32> = (0..(batch * seq_q * dim)).map(|_| negative_value).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| negative_value).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| negative_value).collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    // Create cache and get softmax pipeline
    let mut cache = ResourceCache::new();
    let sdpa_op = cache.get_or_create_sdpa(batch, seq_q, seq_k, dim);
    let softmax_pipeline = context.fused_softmax_pipeline.as_ref().unwrap().clone();

    let result = scaled_dot_product_attention_impl(
        &q_tensor,
        &k_tensor,
        &v_tensor,
        false, // causal
        &mut cache,
        device,
        command_queue,
        &softmax_pipeline,
        sdpa_op.scale,
        &out_tensor,
        &attn_tensor,
    )?;

    let output = result.as_slice();

    // Verify output does not contain infinities or NaNs
    for &val in output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }

    Ok(())
}

#[test]
fn test_sdpa_mixed_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Ensure the fused softmax pipeline is available
    ensure_fused_softmax_pipeline(&mut context)?;

    // Test with mixed extreme values (very large positive and negative)
    let batch = 1;
    let seq_q = 2;
    let seq_k = 2;
    let dim = 2;

    let q_data = vec![1e6f32, -1e6f32, 1e6f32, -1e6f32];
    let k_data = vec![1e6f32, -1e6f32, 1e6f32, -1e6f32];
    let v_data = vec![1e6f32, -1e6f32, 1e6f32, -1e6f32];

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?; // causal = false

    // Synchronize to ensure the SDPA operation is complete before reading values
    context.synchronize();

    let output = result.as_slice();

    // Verify output does not contain infinities or NaNs
    for &val in output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }

    Ok(())
}

#[test]
fn test_sdpa_causal_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Ensure the fused softmax pipeline is available
    ensure_fused_softmax_pipeline(&mut context)?;

    let batch = 1;
    let seq_q = 3;
    let seq_k = 3;
    let dim = 2;

    // Create tensor with very large values
    let large_value = 1e5f32;
    let q_data: Vec<f32> = (0..(batch * seq_q * dim)).map(|_| large_value).collect();
    let k_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| large_value).collect();
    let v_data: Vec<f32> = (0..(batch * seq_k * dim)).map(|_| large_value).collect();

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)?; // causal = true

    // Synchronize to ensure the SDPA operation is complete before reading values
    context.synchronize();

    let output = result.as_slice();

    // Verify output does not contain infinities or NaNs
    for &val in output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }

    Ok(())
}

#[test]
fn test_sdpa_zero_tensors() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Ensure the fused softmax pipeline is available
    ensure_fused_softmax_pipeline(&mut context)?;

    // Test with all zero tensors (edge case)
    let batch = 1;
    let seq_q = 2;
    let seq_k = 2;
    let dim = 2;

    let q_data = vec![0.0f32; batch * seq_q * dim];
    let k_data = vec![0.0f32; batch * seq_k * dim];
    let v_data = vec![0.0f32; batch * seq_k * dim];

    let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &context)?;

    let result = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)?; // causal = false

    // Synchronize to ensure the SDPA operation is complete before reading values
    context.synchronize();

    let output = result.as_slice();

    // Verify output does not contain infinities or NaNs
    for &val in output {
        assert!(val.is_finite(), "Output contains non-finite value: {}", val);
    }

    // For zero tensors, attention should be uniform and output should be average of values
    // (since all attention values are equal, and softmax of identical values is uniform)

    Ok(())
}
