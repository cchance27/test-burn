#![cfg(test)]
use objc2_metal::MTLComputePipelineState as _;

use crate::metallic::kernels::softmax::SoftmaxOp;
use crate::metallic::{Context, MetalError, Tensor};

// CPU-based softmax for golden testing
fn cpu_softmax(input: &[f32], seq_q: usize, seq_k: usize, causal: bool) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let mut row = input[row_start..row_end].to_vec();

        if causal {
            // Apply causal mask: set future elements to -INFINITY
            ((i + 1)..seq_k).for_each(|j| {
                row[j] = -f32::INFINITY;
            });
        }

        let max_val = row.iter().fold(-f32::INFINITY, |acc, &x| x.max(acc));
        let mut exp_sum = 0.0;
        (0..seq_k).for_each(|j| {
            let exp_val = (row[j] - max_val).exp();
            row[j] = exp_val;
            exp_sum += exp_val;
        });

        for j in 0..seq_k {
            output[row_start + j] = row[j] / exp_sum;
        }
    }
    output
}

fn softmax_rows_total(attn_tensor: &Tensor, seq_k: usize) -> u32 {
    if seq_k == 0 {
        0
    } else {
        (attn_tensor.len() / seq_k) as u32
    }
}

#[test]
fn test_softmax_golden_non_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    let seq_q = 2;
    let seq_k = 4;
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    // Apply softmax using the new kernel system (in-place operation)
    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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
fn test_softmax_golden_causal() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    let seq_q = 2;
    let seq_k = 4;
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, true);

    // Apply softmax using the new kernel system (in-place operation) with causal masking
    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        1,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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

    // Verify row sums to ~1.0 and masked elements are effectively zeroed
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let row_slice = &metal_output[row_start..row_end];
        let row_sum: f32 = row_slice.iter().sum();

        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );

        // Check masked elements
        ((i + 1)..seq_k).for_each(|j| {
            assert!(
                row_slice[j].abs() < 1e-5,
                "Masked element at [{}, {}] not zeroed: got {:.6}",
                i,
                j,
                row_slice[j]
            );
        });
    }

    Ok(())
}

// Additional Softmax Tests from src/metallic/tests/softmax_extremes_test.rs

#[test]
fn test_softmax_extremes_large_positive_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 3;
    let seq_k = 4;

    // Create input with large positive values
    let input_data = vec![
        1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0,
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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
fn test_softmax_extremes_large_negative_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 2;
    let seq_k = 3;

    // Create input with large negative values
    let input_data = vec![-1000.0, -2000.0, -3000.0, -4000.0, -5000.0, -6000.0];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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
fn test_softmax_extremes_identical_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 2;
    let seq_k = 4;

    // Create input with identical values in each row
    let input_data = vec![5.0, 5.0, 5.0, 5.0, -10.0, -10.0, -10.0, -10.0];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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

    // For identical values, each element should be 1/seq_k
    let expected_val = 1.0 / seq_k as f32;
    (0..(seq_q * seq_k)).for_each(|i| {
        assert!(
            (metal_output[i] - expected_val).abs() < 1e-5,
            "Element {} should be {:.6}, got {:.6}",
            i,
            expected_val,
            metal_output[i]
        );
    });

    Ok(())
}

#[test]
fn test_softmax_extremes_single_large_outlier() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 2;
    let seq_k = 5;

    // Create input with one very large outlier
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 10000.0, // Large outlier at the end
        -1.0, -2.0, -3.0, -4.0, 5.0, // Moderate outlier at the end
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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

    // For the first row, the outlier should dominate (be close to 1.0)
    assert!(
        (metal_output[4] - 1.0).abs() < 1e-3,
        "Outlier element should be close to 1.0, got {:.6}",
        metal_output[4]
    );

    // Other elements in the first row should be close to 0
    (0..4).for_each(|i| {
        assert!(
            metal_output[i].abs() < 1e-3,
            "Non-outlier element {} should be close to 0, got {:.6}",
            i,
            metal_output[i]
        );
    });

    Ok(())
}

#[test]
fn test_softmax_extremes_causal_with_extremes() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 3;
    let seq_k = 3;

    // Create input with extreme values and test causal masking
    let input_data = vec![1000.0, -1000.0, 5000.0, -2000.0, 3000.0, -4000.0, 100.0, 200.0, 300.0];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, true);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        1,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Verify row sums to ~1.0 and masked elements are effectively zeroed
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let row_slice = &metal_output[row_start..row_end];
        let row_sum: f32 = row_slice.iter().sum();

        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );

        // Check masked elements
        ((i + 1)..seq_k).for_each(|j| {
            assert!(
                row_slice[j].abs() < 1e-5,
                "Masked element at [{}, {}] not zeroed: got {:.6}",
                i,
                j,
                row_slice[j]
            );
        });
    }

    Ok(())
}

// Additional Softmax Tests from src/metallic/tests/softmax_extremes_additional_test.rs

// CPU-based softmax for golden testing with special handling for infinity values
fn cpu_softmax_with_infinity(input: &[f32], seq_q: usize, seq_k: usize, causal: bool) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let mut row = input[row_start..row_end].to_vec();

        if causal {
            // Apply causal mask: set future elements to -INFINITY
            ((i + 1)..seq_k).for_each(|j| {
                row[j] = -f32::INFINITY;
            });
        }

        // Handle infinity - if any value is +inf, it gets 1.0 and others 0.0
        if row.contains(&f32::INFINITY) {
            let mut total_inf = 0.0;
            for &val in &row {
                if val == f32::INFINITY {
                    total_inf += 1.0;
                }
            }
            for j in 0..seq_k {
                output[row_start + j] = if row[j] == f32::INFINITY { 1.0 / total_inf } else { 0.0 };
            }
            continue;
        }

        let max_val = row.iter().fold(-f32::INFINITY, |acc, &x| x.max(acc));
        let mut exp_sum = 0.0;
        (0..seq_k).for_each(|j| {
            let exp_val = (row[j] - max_val).exp();
            row[j] = exp_val;
            exp_sum += exp_val;
        });

        for j in 0..seq_k {
            output[row_start + j] = row[j] / exp_sum;
        }
    }
    output
}

#[test]
fn test_softmax_extremes_underflow_scenarios() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 2;
    let seq_k = 4;

    // Test with extremely small values that may cause underflow issues
    let input_data = vec![
        -10000.0, -9999.0, -10001.0, -10002.0, // Very negative values that could cause exp to underflow
        -10000.0, -10000.0, -10000.0, -10000.0, // Identical very negative values
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax_with_infinity(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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
fn test_softmax_extremes_infinity_scenarios() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 2;
    let seq_k = 4;

    // Test with infinity values
    let input_data = vec![
        f32::INFINITY,
        1.0,
        2.0,
        3.0, // One infinity should result in 1.0 at that position
        1.0,
        f32::NEG_INFINITY,
        2.0,
        3.0, // NEG_INFINITY should not affect softmax significantly
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax_with_infinity(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

        // Check for NaN or Infinity - the output should be finite after softmax
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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
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

    // For the first row, with infinity at position 0, it should be 1.0 and others 0.0
    assert!(
        (metal_output[0] - 1.0).abs() < 1e-5,
        "First row should have 1.0 at position 0, got {}",
        metal_output[0]
    );
    (1..4).for_each(|j| {
        assert!(
            metal_output[j].abs() < 1e-5,
            "First row should have 0.0 at position {}, got {}",
            j,
            metal_output[j]
        );
    });

    Ok(())
}

#[test]
fn test_softmax_extremes_nan_scenarios() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 1;
    let seq_k = 4;

    // Test with NaN values - this can cause issues
    let input_data = vec![
        f32::NAN,
        1.0,
        2.0,
        3.0, // NaN in the input
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax_with_infinity(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    // When input contains NaN, output should also contain NaN
    for i in 0..seq_k {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;

        // Both outputs should be NaN
        assert!(
            metal_val.is_nan(),
            "Metal output should be NaN at index {} when input contains NaN, got {}",
            i,
            metal_val
        );
        assert!(
            cpu_val.is_nan(),
            "CPU output should be NaN at index {} when input contains NaN, got {}",
            i,
            cpu_val
        );
    }

    Ok(())
}

#[test]
fn test_softmax_extremes_large_sequences() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 1;
    let seq_k = 2048; // Large sequence length

    // Create input with varied values including some extreme ones
    let mut input_data = Vec::with_capacity(seq_k);
    for i in 0..seq_k {
        if i % 500 == 0 {
            input_data.push(1000.0); // Some extreme values
        } else {
            input_data.push((i % 100) as f32); // Regular values
        }
    }

    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax_with_infinity(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Verify row sums to ~1.0
    let row_sum: f32 = metal_output.iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-5, "Row sum mismatch: expected ~1.0, got {:.6}", row_sum);

    Ok(())
}

#[test]
fn test_softmax_extremes_very_large_positive_and_negative_mixed() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 1;
    let seq_k = 8;

    // Mix of very large positive and negative values
    let input_data = vec![
        1e10,  // Very large positive
        -1e10, // Very large negative
        1e9,   // Large positive
        -1e9,  // Large negative
        5.0,   // Moderate
        -5.0,  // Moderate negative
        0.0,   // Zero
        100.0, // Large
    ];
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax_with_infinity(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    for i in 0..input_data.len() {
        let metal_val = metal_output[i] as f64;
        let cpu_val = cpu_output[i] as f64;
        let diff = (metal_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 { diff / cpu_val.abs() } else { diff };

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
            "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            cpu_val,
            diff,
            rel_err
        );
    }

    // Verify row sums to ~1.0
    let row_sum: f32 = metal_output.iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-5, "Row sum mismatch: expected ~1.0, got {:.6}", row_sum);

    // The largest positive value (1e10) should dominate and be close to 1.0
    assert!(
        (metal_output[0] - 1.0).abs() < 1e-6,
        "Largest value should dominate softmax, expected ~1.0, got {}",
        metal_output[0]
    );

    Ok(())
}

// Additional Softmax Tests from src/metallic/tests/softmax_irregular_test.rs

#[test]
fn test_softmax_irregular_sizes_1() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 7;
    let seq_k = 13;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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
fn test_softmax_irregular_sizes_2() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 31;
    let seq_k = 257;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.01).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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
fn test_softmax_causal_irregular_sizes() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 5;
    let seq_k = 9;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.2).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, true);

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        1,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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

    // Verify row sums to ~1.0 and masked elements are effectively zeroed
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let row_slice = &metal_output[row_start..row_end];
        let row_sum: f32 = row_slice.iter().sum();

        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );

        // Check masked elements
        ((i + 1)..seq_k).for_each(|j| {
            assert!(
                row_slice[j].abs() < 1e-5,
                "Masked element at [{}, {}] not zeroed: got {:.6}",
                i,
                j,
                row_slice[j]
            );
        });
    }

    Ok(())
}

#[test]
fn test_softmax_causal_large_irregular() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    let seq_q = 17;
    let seq_k = 93;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.05).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, true); // Using causal=true

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        1,
        0,
    ))?;
    context.synchronize();

    let metal_output = result.as_slice();

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

    // Verify row sums to ~1.0 and masked elements are effectively zeroed
    for i in 0..seq_q {
        let row_start = i * seq_k;
        let row_end = (i + 1) * seq_k;
        let row_slice = &metal_output[row_start..row_end];
        let row_sum: f32 = row_slice.iter().sum();

        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row {} sum mismatch: expected ~1.0, got {:.6}",
            i,
            row_sum
        );

        // Check masked elements
        ((i + 1)..seq_k).for_each(|j| {
            assert!(
                row_slice[j].abs() < 1e-5,
                "Masked element at [{}, {}] not zeroed: got {:.6}",
                i,
                j,
                row_slice[j]
            );
        });
    }

    Ok(())
}

// Additional Softmax Tests from src/metallic/tests/softmax_threadgroup_test.rs

#[test]
fn test_softmax_threadgroup_execution_width() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Get the pipeline through the new kernel system
    let pipeline = context
        .kernel_manager
        .get_pipeline(crate::metallic::kernels::KernelFunction::FusedSoftmax, &context.device)?;

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
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    // Verify the operation completed successfully
    let metal_output = result.as_slice();
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

    // Test with a larger sequence length that requires many threads per threadgroup
    let seq_q = 2;
    let seq_k = 128; // Larger than typical threadgroup size
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.05).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    // Verify the operation completed successfully
    let metal_output = result.as_slice();
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

    // Test with a very large sequence length
    let seq_q = 1;
    let seq_k = 512; // Very large sequence length
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.01).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    // Verify the operation completed successfully
    let metal_output = result.as_slice();
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
fn test_softmax_threadgroup_multiple_rows() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // Test with multiple rows to ensure threadgroup handling works across rows
    let seq_q = 16; // Multiple rows
    let seq_k = 64; // Reasonable size per row
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.02).collect();
    let input_tensor = Tensor::create_tensor_from_slice(&input_data, vec![seq_q, seq_k], &context)?; // Reshape to 2D as expected by kernel
    let attn_tensor = input_tensor.clone();

    let result = context.call::<SoftmaxOp>((
        &attn_tensor,
        softmax_rows_total(&attn_tensor, seq_k),
        seq_q as u32,
        seq_k as u32,
        0,
        0,
    ))?;
    context.synchronize();

    // Verify the operation completed successfully
    let metal_output = result.as_slice();
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

#[test]
fn test_softmax_logic() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;
    // Create a simple test tensor [2, 3] with values that will produce recognizable softmax results
    let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // Two rows to softmax independently
    let attn = Tensor::create_tensor_from_slice(&input_data, vec![2, 3], &ctx)?;
    // Apply softmax with no causal masking (causal=0)
    let result = ctx.call::<SoftmaxOp>((&attn, softmax_rows_total(&attn, 3), 2, 3, 0, 0))?;
    // Check that each row sums to approximately 1 (property of softmax)
    let result_slice = result.as_slice();
    let row1_sum: f32 = result_slice[0..3].iter().sum();
    let row2_sum: f32 = result_slice[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be 1.0, got {}", row1_sum);
    assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 sum should be 1.0, got {}", row2_sum);
    Ok(())
}
