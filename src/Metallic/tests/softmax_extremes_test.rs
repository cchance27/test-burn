use crate::metallic::softmax::SoftmaxOperation;

use super::*;

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

#[test]
fn test_softmax_extremes_large_positive_values() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 3;
    let seq_k = 4;

    // Create input with large positive values
    let input_data = vec![
        1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0,
        12000.0,
    ];
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = attn_tensor.as_slice();

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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 2;
    let seq_k = 3;

    // Create input with large negative values
    let input_data = vec![-1000.0, -2000.0, -3000.0, -4000.0, -5000.0, -6000.0];
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = attn_tensor.as_slice();

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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 2;
    let seq_k = 4;

    // Create input with identical values in each row
    let input_data = vec![5.0, 5.0, 5.0, 5.0, -10.0, -10.0, -10.0, -10.0];
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = attn_tensor.as_slice();

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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 2;
    let seq_k = 5;

    // Create input with one very large outlier
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 10000.0, // Large outlier at the end
        -1.0, -2.0, -3.0, -4.0, 5.0, // Moderate outlier at the end
    ];
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 0,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = attn_tensor.as_slice();

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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 3;
    let seq_k = 3;

    // Create input with extreme values and test causal masking
    let input_data = vec![
        1000.0, -1000.0, 5000.0, -2000.0, 3000.0, -4000.0, 100.0, 200.0, 300.0,
    ];
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, true);

    let sm_op = SoftmaxOperation {
        attn_buf: attn_tensor.buf.clone(),
        attn_offset: attn_tensor.offset,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        causal: 1,
        pipeline: context.fused_softmax_pipeline.as_ref().unwrap().clone(),
    };

    let command_buffer = context.command_queue.commandBuffer().unwrap();
    let mut cache = ResourceCache::new();
    sm_op.encode(&command_buffer, &mut cache)?;
    command_buffer.commit();
    unsafe {
        command_buffer.waitUntilCompleted();
    }

    let metal_output = attn_tensor.as_slice();

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
