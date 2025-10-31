use crate::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage, kernels::rmsnorm::RMSNormOp};

#[test]
fn test_rmsnorm_logic() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Create input tensor with shape [2, 4] (2 rows, 4 features each)
    let input_data = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let input = Tensor::new(vec![2, 4], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

    // Create gamma (weight) tensor
    let gamma_data = vec![1.0, 1.0, 1.0, 1.0];
    let gamma = Tensor::new(vec![4], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&gamma_data))?;

    let feature_dim = 4u32;

    // Use the kernel via the generic `call` method.
    let result_tensor = ctx.call::<RMSNormOp>((input, gamma, feature_dim))?;

    // Verify the output shape matches input shape
    assert_eq!(result_tensor.dims(), &[2, 4]);

    Ok(())
}

// Additional RMSNorm Tests from src/metallic/tests/rmsnorm_test.rs

// CPU-based RMSNorm for golden testing (matches EPS=1e-5 used in kernel)
fn cpu_rmsnorm(input: &[f32], feature_dim: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    let eps = eps as f64;
    let mut output = vec![0.0f32; input.len()];
    let num_rows = input.len() / feature_dim;

    for row in 0..num_rows {
        let row_start = row * feature_dim;
        let row_end = row_start + feature_dim;
        let row_data = &input[row_start..row_end];

        let sum_sq: f64 = row_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let rms = (sum_sq / feature_dim as f64 + eps).sqrt();

        (0..feature_dim).for_each(|f| {
            let idx = row_start + f;
            output[idx] = ((input[idx] as f64) / rms * gamma[f] as f64) as f32;
        });
    }

    output
}

#[test]
fn test_rmsnorm_basic() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

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

    let dims = vec![batch_size, seq_len, feature_dim];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let gamma_tensor = Tensor::new(
        vec![feature_dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&gamma_data),
    )?;

    let output_tensor = context.call::<RMSNormOp>((input_tensor, gamma_tensor, feature_dim as u32))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();
    let cpu_output = cpu_rmsnorm(&input_data, feature_dim, &gamma_data, 1e-5);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
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
fn test_rmsnorm_numerical_stability() -> Result<(), MetalError> {
    let mut context = Context::<F32Element>::new()?;

    let feature_dim = 4;
    let batch_size = 1;
    let seq_len = 1;

    // Test with large values that could cause overflow in sum-of-squares
    let input_data = vec![
        1e3, 2e3, 3e3, 4e3, // Large positive values
    ];

    let gamma_data = vec![1.0, 1.0, 1.0, 1.0];

    let dims = vec![batch_size, seq_len, feature_dim];
    let input_tensor = Tensor::new(dims.clone(), TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&input_data))?;
    let gamma_tensor = Tensor::new(
        vec![feature_dim],
        TensorStorage::Dedicated(&context),
        TensorInit::CopyFrom(&gamma_data),
    )?;

    let output_tensor = context.call::<RMSNormOp>((input_tensor, gamma_tensor, feature_dim as u32))?;
    context.synchronize();

    let metal_output = output_tensor.as_slice();

    // Check that the output values are finite (not NaN or infinity)
    for &val in metal_output {
        assert!(val.is_finite(), "RMSNorm output contains non-finite value: {}", val);
    }

    // Compare with CPU implementation
    let cpu_output = cpu_rmsnorm(&input_data, feature_dim, &gamma_data, 1e-5);

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
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
