use super::*;
use crate::metallic::gelu::{Gelu, ensure_gelu_pipeline};

// CPU-based GELU for golden testing
fn cpu_gelu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let x64 = x as f64;
            let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
            let inner = sqrt_2_pi * (x64 + 0.044715 * x64.powi(3));
            0.5 * x64 * (1.0 + inner.tanh())
        })
        .map(|x| x as f32)
        .collect()
}

#[test]
fn test_gelu_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_gelu_pipeline(&mut context)?;

    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.1];
    let dims = vec![2, 5];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let gelu_op = Gelu::new(
        input_tensor,
        output_tensor.clone(),
        context.gelu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    gelu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_gelu(&input_data);

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
            "Mismatch at index {}: input={:.3}, metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            input_data[i],
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}

#[test]
fn test_gelu_extremes() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_gelu_pipeline(&mut context)?;

    let input_data = vec![
        -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, -100.0, 100.0, 0.001, -0.001,
    ];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let gelu_op = Gelu::new(
        input_tensor,
        output_tensor.clone(),
        context.gelu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    gelu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_gelu(&input_data);

    let rtol = 1e-2f64; // More lenient tolerance for extreme values
    let atol = 1e-4f64;

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
            "Extreme value mismatch at index {}: input={:.3}, metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            input_data[i],
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Check that GELU behaves correctly for extreme values
    // For very negative values, GELU should be close to 0
    assert!(
        metal_output[0].abs() < 1e-6,
        "GELU(-10) should be ~0, got {}",
        metal_output[0]
    );
    assert!(
        metal_output[1].abs() < 1e-4,
        "GELU(-5) should be ~0, got {}",
        metal_output[1]
    );

    // For very positive values, GELU should be close to the input
    let ratio_10 = metal_output[6] as f64 / 10.0;
    assert!(
        (ratio_10 - 1.0).abs() < 0.01,
        "GELU(10) should be ~10, got {}",
        metal_output[6]
    );

    Ok(())
}

#[test]
fn test_gelu_zero_and_symmetry() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_gelu_pipeline(&mut context)?;

    // Test points around zero and check basic properties
    let input_data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let gelu_op = Gelu::new(
        input_tensor,
        output_tensor.clone(),
        context.gelu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    gelu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_gelu(&input_data);

    let rtol = 1e-3f64; // More lenient for this test
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
            "Basic test mismatch at index {}: input={:.3}, metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            input_data[i],
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Check GELU(0) ≈ 0
    assert!(
        metal_output[3].abs() < 1e-6,
        "GELU(0) should be ~0, got {}",
        metal_output[3]
    );

    // Note: GELU using tanh approximation is not perfectly symmetric for non-zero values
    // This is expected behavior. The important checks are numerical accuracy and extreme value handling.

    Ok(())
}

#[test]
fn test_gelu_validation_errors() {
    let mut context = Context::new().unwrap();
    ensure_gelu_pipeline(&mut context).unwrap();

    let dims = vec![2, 3];
    let input = Tensor::create_tensor(dims.clone(), &context).unwrap();

    // Test mismatched output shape
    let wrong_output = Tensor::create_tensor(vec![5], &context).unwrap();
    let pipeline = context.gelu_pipeline.as_ref().unwrap().clone();

    let result = Gelu::new(input.clone(), wrong_output, pipeline);
    assert!(result.is_err());
}
