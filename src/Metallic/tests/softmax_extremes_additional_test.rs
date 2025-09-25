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

        // Handle infinity - if any value is +inf, it gets 1.0 and others 0.0
        if row.contains(&f32::INFINITY) {
            let mut total_inf = 0.0;
            for &val in &row {
                if val == f32::INFINITY {
                    total_inf += 1.0;
                }
            }
            for j in 0..seq_k {
                output[row_start + j] = if row[j] == f32::INFINITY {
                    1.0 / total_inf
                } else {
                    0.0
                };
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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 2;
    let seq_k = 4;

    // Test with extremely small values that may cause underflow issues
    let input_data = vec![
        -10000.0, -9999.0, -10001.0,
        -10002.0, // Very negative values that could cause exp to underflow
        -10000.0, -10000.0, -10000.0, -10000.0, // Identical very negative values
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
fn test_softmax_extremes_infinity_scenarios() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;
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
    let dims = vec![1, seq_q, seq_k];
    let attn_tensor = Tensor::create_tensor_from_slice(&input_data, dims.clone(), &context)?;

    let cpu_output = cpu_softmax(&input_data, seq_q, seq_k, false);
    context.synchronize();

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
    context.synchronize();

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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 1;
    let seq_k = 4;

    // Test with NaN values - this can cause issues
    let input_data = vec![
        f32::NAN,
        1.0,
        2.0,
        3.0, // NaN in the input
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
    ensure_fused_softmax_pipeline(&mut context)?;
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
    let row_sum: f32 = metal_output.iter().sum();
    assert!(
        (row_sum - 1.0).abs() < 1e-5,
        "Row sum mismatch: expected ~1.0, got {:.6}",
        row_sum
    );

    Ok(())
}

#[test]
fn test_softmax_extremes_very_large_positive_and_negative_mixed() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;
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
    let row_sum: f32 = metal_output.iter().sum();
    assert!(
        (row_sum - 1.0).abs() < 1e-5,
        "Row sum mismatch: expected ~1.0, got {:.6}",
        row_sum
    );

    // The largest positive value (1e10) should dominate and be close to 1.0
    assert!(
        (metal_output[0] - 1.0).abs() < 1e-6,
        "Largest value should dominate softmax, expected ~1.0, got {}",
        metal_output[0]
    );

    Ok(())
}
