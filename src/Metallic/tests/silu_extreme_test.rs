use super::*;
use crate::metallic::silu::{Silu, ensure_silu_pipeline};

// CPU-based SiLU for golden testing
fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            if x > 50.0 {
                x // Clamped to x for numerical stability
            } else if x < -50.0 {
                0.0 // Clamped to 0.0 for numerical stability
            } else {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
        })
        .collect()
}

#[test]
fn test_silu_extreme_positive_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    // Create input with extremely large positive values
    let input_data = vec![100.0f32, 1000.0f32, 10000.0f32, 1e6f32];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let cpu_output = cpu_silu(&input_data);

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();

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

        // Check for NaN or Infinity
        assert!(
            metal_val.is_finite(),
            "Metal output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_finite(),
            "CPU output contains non-finite value at index {}: {}",
            i,
            cpu_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "SiLU mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // For extremely large positive values, SiLU should approach the input value (x)
    for (i, &input_val) in input_data.iter().enumerate() {
        if input_val > 50.0 {
            // When x is > 50, SiLU(x) should be approximately x due to clamping
            let expected_output = input_val as f64;
            let actual_output = metal_output[i] as f64;
            let diff = (actual_output - expected_output).abs();
            assert!(
                diff < 1e-6,
                "For large positive input {}, output should be approximately {}, got {}",
                input_val,
                expected_output,
                actual_output
            );
        }
    }

    Ok(())
}

#[test]
fn test_silu_extreme_negative_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    // Create input with extremely large negative values
    let input_data = vec![-100.0f32, -1000.0f32, -10000.0f32, -1e6f32];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let cpu_output = cpu_silu(&input_data);

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();

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

        // Check for NaN or Infinity
        assert!(
            metal_val.is_finite(),
            "Metal output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_finite(),
            "CPU output contains non-finite value at index {}: {}",
            i,
            cpu_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "SiLU mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // For extremely large negative values, SiLU should approach 0.0 due to clamping
    for (i, &input_val) in input_data.iter().enumerate() {
        if input_val < -50.0 {
            // When x is < -50, SiLU(x) should be approximately 0.0 due to clamping
            let expected_output = 0.0f64;
            let actual_output = metal_output[i] as f64;
            let diff = (actual_output - expected_output).abs();
            assert!(
                diff < 1e-6,
                "For large negative input {}, output should be approximately {}, got {}",
                input_val,
                expected_output,
                actual_output
            );
        }
    }

    Ok(())
}

#[test]
fn test_silu_mixed_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    // Create input with mixed extreme values
    let input_data = vec![
        1e8f32,   // Very large positive
        -1e8f32,  // Very large negative
        1e-8f32,  // Very small positive
        -1e-8f32, // Very small negative
        49.9f32,  // Just below positive clamp threshold
        -49.9f32, // Just above negative clamp threshold
        50.1f32,  // Just above positive clamp threshold
        -50.1f32, // Just below negative clamp threshold
    ];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let cpu_output = cpu_silu(&input_data);

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();

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

        // Check for NaN or Infinity
        assert!(
            metal_val.is_finite(),
            "Metal output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_finite(),
            "CPU output contains non-finite value at index {}: {}",
            i,
            cpu_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "SiLU mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
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
fn test_silu_edge_values_around_thresholds() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    // Create input values around the clamping thresholds to test edge cases
    let input_data = vec![
        49.0f32,  // Below positive threshold
        49.5f32,  // Close to positive threshold
        49.9f32,  // Very close to positive threshold
        50.0f32,  // Exactly at positive threshold
        50.1f32,  // Just above positive threshold
        51.0f32,  // Above positive threshold
        -49.0f32, // Above negative threshold
        -49.5f32, // Close to negative threshold
        -49.9f32, // Very close to negative threshold
        -50.0f32, // Exactly at negative threshold
        -50.1f32, // Just below negative threshold
        -51.0f32, // Below negative threshold
    ];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let cpu_output = cpu_silu(&input_data);

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();

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

        // Check for NaN or Infinity
        assert!(
            metal_val.is_finite(),
            "Metal output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_finite(),
            "CPU output contains non-finite value at index {}: {}",
            i,
            cpu_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "SiLU mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Specifically check values around thresholds
    for (i, &input_val) in input_data.iter().enumerate() {
        if input_val > 50.0 {
            // Should be clamped to the input value
            assert!(
                (metal_output[i] - input_val).abs() < 1e-6,
                "Value {} above threshold should be clamped to input {}, got {}",
                input_val,
                input_val,
                metal_output[i]
            );
        } else if input_val < -50.0 {
            // Should be clamped to 0.0
            assert!(
                metal_output[i].abs() < 1e-6,
                "Value {} below threshold should be clamped to 0.0, got {}",
                input_val,
                metal_output[i]
            );
        }
    }

    Ok(())
}

#[test]
fn test_silu_large_tensor_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_silu_pipeline(&mut context)?;

    // Create a larger tensor with extreme values
    let size = 1000;
    let mut input_data = Vec::with_capacity(size);
    for i in 0..size {
        match i % 4 {
            0 => input_data.push(1e5f32),  // Large positive
            1 => input_data.push(-1e5f32), // Large negative
            2 => input_data.push(0.1f32),  // Small positive
            _ => input_data.push(-0.1f32), // Small negative
        }
    }

    let dims = vec![size];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;
    let output_tensor = Tensor::create_tensor(dims.clone(), &context)?;

    let cpu_output = cpu_silu(&input_data);

    let silu_op = Silu::new(
        input_tensor,
        output_tensor.clone(),
        context.silu_pipeline.as_ref().unwrap().clone(),
    )?;

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    silu_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = output_tensor.as_slice();

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

        // Check for NaN or Infinity
        assert!(
            metal_val.is_finite(),
            "Metal output contains non-finite value at index {}: {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_finite(),
            "CPU output contains non-finite value at index {}: {}",
            i,
            cpu_val
        );

        assert!(
            diff <= atol || rel_err <= rtol,
            "SiLU mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    Ok(())
}
