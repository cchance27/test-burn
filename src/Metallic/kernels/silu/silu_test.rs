#![cfg(test)]
use crate::metallic::kernels::silu::SiluOp;
use crate::metallic::{Context, MetalError, Tensor, TensorInit, TensorStorage};

// CPU SiLU
fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let x64 = x as f64;
            let sig = 1.0 / (1.0 + (-x64).exp());
            (x64 * sig) as f32
        })
        .collect()
}

#[test]
fn test_silu_basic() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -50.0, 50.0];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_silu(&input_data);

    for (i, (metal_val, cpu_val)) in metal_output.iter().zip(cpu_output.iter()).enumerate() {
        let diff = (metal_val - cpu_val).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at index {}: metal={}, cpu={}, diff={}",
            i,
            metal_val,
            cpu_val,
            diff
        );
    }

    Ok(())
}

#[test]
fn test_silu_numerical_stability() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    

    // Test with extreme values that could cause overflow in exp computation
    let input_data = vec![
        -100.0, -50.0, -20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0, 50.0, 100.0,
    ];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();

    // Check that the output values are finite (not NaN or infinity)
    for &val in metal_output {
        assert!(
            val.is_finite(),
            "SiLU output contains non-finite value: {}",
            val
        );
    }

    let cpu_output = cpu_silu(&input_data);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for i in 0..input_data.len() {
        let m = metal_output[i] as f64;
        let c = cpu_output[i] as f64;
        let diff = (m - c).abs();
        let rel = if c.abs() > 1e-8 { diff / c.abs() } else { diff };
        assert!(
            diff <= atol || rel <= rtol,
            "Mismatch at {}: metal={}, cpu={}, diff={}",
            i,
            m,
            c,
            diff
        );
    }

    Ok(())
}

// Additional SiLU Tests from src/metallic/tests/silu_extreme_test.rs

// CPU-based SiLU for golden testing
fn cpu_silu_extreme(input: &[f32]) -> Vec<f32> {
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

    // Create input with extremely large positive values
    let input_data = vec![100.0f32, 1000.0f32, 10000.0f32, 1e6f32];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let cpu_output = cpu_silu_extreme(&input_data);

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

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

    // Create input with extremely large negative values
    let input_data = vec![-100.0f32, -1000.0f32, -10000.0f32, -1e6f32];
    let dims = vec![input_data.len()];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let cpu_output = cpu_silu_extreme(&input_data);

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

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
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let cpu_output = cpu_silu_extreme(&input_data);

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

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

    // Test around key thresholds in the SiLU implementation
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
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let cpu_output = cpu_silu_extreme(&input_data);

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

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
fn test_silu_large_tensor_extreme_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Create a larger tensor with mixed extreme values
    let size = 1000;
    let mut input_data = Vec::with_capacity(size);
    for i in 0..size {
        match i % 4 {
            0 => input_data.push(i as f32 * 0.1),        // gradually increasing positive
            1 => input_data.push(-(i as f32) * 0.1),     // gradually decreasing negative
            2 => input_data.push(1000.0),                 // large positive
            3 => input_data.push(-1000.0),                // large negative
            _ => unreachable!(),
        }
    }

    let dims = vec![input_data.len()];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;

    let cpu_output = cpu_silu_extreme(&input_data);

    let output_tensor = context.call::<SiluOp>(input_tensor)?;
    context.synchronize();

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