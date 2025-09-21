use super::*;
use crate::metallic::layernorm::{LayerNorm, ensure_layernorm_pipeline};

// CPU-based layer normalization for golden testing
fn cpu_layernorm(
    input: &[f32],
    feature_dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let eps = eps as f64;
    let mut output = vec![0.0; input.len()];

    let num_rows = input.len() / feature_dim;

    for row in 0..num_rows {
        let row_start = row * feature_dim;
        let row_end = (row + 1) * feature_dim;
        let row_data = &input[row_start..row_end];

        // Calculate mean
        let sum: f64 = row_data.iter().map(|&x| x as f64).sum();
        let mean = sum / feature_dim as f64;

        // Calculate variance
        let sum_sq: f64 = row_data.iter().map(|&x| (x as f64 - mean).powi(2)).sum();
        let var = sum_sq / feature_dim as f64;

        // Normalize and apply affine transformation
        for f in 0..feature_dim {
            let idx = row_start + f;
            let x = input[idx] as f64;
            let normalized = (x - mean) / (var + eps).sqrt();
            output[idx] = (normalized * gamma[f] as f64 + beta[f] as f64) as f32;
        }
    }

    output
}

#[test]
fn test_layernorm_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_layernorm_pipeline(&mut context)?;

    let feature_dim = 4;
    let batch_size = 2;
    let seq_len = 3;

    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
        2.0, 4.0, 6.0, 8.0, // row 3
        1.5, 3.5, 5.5, 7.5, // row 4
        0.5, 2.5, 4.5, 6.5, // row 5
    ];

    let gamma_data = vec![1.0, 1.1, 1.2, 1.3];
    let beta_data = vec![0.1, 0.2, 0.3, 0.4];

    let dims = vec![batch_size, seq_len, feature_dim];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;
    let gamma_tensor = Tensor::create_tensor_from_slice(&gamma_data, vec![feature_dim], &context)?;
    let beta_tensor = Tensor::create_tensor_from_slice(&beta_data, vec![feature_dim], &context)?;

    let layernorm_op = LayerNorm::new(
        input_tensor,
        output_tensor.clone(),
        gamma_tensor,
        beta_tensor,
        feature_dim as u32,
        context.layernorm_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    layernorm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_layernorm(&input_data, feature_dim, &gamma_data, &beta_data, 1e-5);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 {
            diff / cpu_val.abs()
        } else {
            diff
        };
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
fn test_layernorm_identity_transform() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_layernorm_pipeline(&mut context)?;

    let feature_dim = 3;
    let batch_size = 1;
    let seq_len = 2;

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 1.0, 1.0]; // Identity scaling
    let beta_data = vec![0.0, 0.0, 0.0]; // No shift

    let dims = vec![batch_size, seq_len, feature_dim];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;
    let gamma_tensor = Tensor::create_tensor_from_slice(&gamma_data, vec![feature_dim], &context)?;
    let beta_tensor = Tensor::create_tensor_from_slice(&beta_data, vec![feature_dim], &context)?;

    let layernorm_op = LayerNorm::new(
        input_tensor,
        output_tensor.clone(),
        gamma_tensor,
        beta_tensor,
        feature_dim as u32,
        context.layernorm_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    layernorm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_layernorm(&input_data, feature_dim, &gamma_data, &beta_data, 1e-5);

    // With identity transform, check that the normalized values are reasonable
    let rtol = 1e-3f64;
    let atol = 1e-5f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 {
            diff / cpu_val.abs()
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Identity transform mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
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
fn test_layernorm_validation_errors() {
    let mut context = Context::new().unwrap();
    ensure_layernorm_pipeline(&mut context).unwrap();

    let dims = vec![2, 3, 4];
    let input = Tensor::create_tensor(dims.clone(), &context).unwrap();
    let output = Tensor::create_tensor(dims.clone(), &context).unwrap();
    let gamma = Tensor::create_tensor(vec![4], &context).unwrap();
    let beta = Tensor::create_tensor(vec![4], &context).unwrap();

    // Use the layernorm pipeline for validation testing
    let pipeline = context.layernorm_pipeline.as_ref().unwrap().clone();

    // Test mismatched feature dimension
    let result = LayerNorm::new(
        input.clone(),
        output.clone(),
        gamma.clone(),
        beta.clone(),
        5, // Wrong feature dimension
        pipeline.clone(),
    );
    assert!(result.is_err());

    // Test mismatched gamma shape
    let wrong_gamma = Tensor::create_tensor(vec![3], &context).unwrap();
    let result = LayerNorm::new(
        input.clone(),
        output.clone(),
        wrong_gamma,
        beta.clone(),
        4,
        pipeline.clone(),
    );
    assert!(result.is_err());

    // Test mismatched output shape
    let wrong_output = Tensor::create_tensor(vec![2, 2, 5], &context).unwrap();
    let result = LayerNorm::new(
        input.clone(),
        wrong_output,
        gamma.clone(),
        beta.clone(),
        4,
        pipeline,
    );
    assert!(result.is_err());
}
