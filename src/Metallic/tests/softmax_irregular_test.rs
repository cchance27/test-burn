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
fn test_softmax_irregular_sizes_1() -> Result<(), MetalError> {
    let mut context = Context::new()?;
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 7;
    let seq_k = 13;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.1).collect();
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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 31;
    let seq_k = 257;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.01).collect();
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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 5;
    let seq_k = 9;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.2).collect();
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
    ensure_fused_softmax_pipeline(&mut context)?;
    let seq_q = 17;
    let seq_k = 129;
    let input_data: Vec<f32> = (0..(seq_q * seq_k)).map(|i| (i as f32) * 0.05).collect();
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
