#![cfg(test)]

use super::*;

fn cpu_repeat_kv_heads(
    input: &[f32],
    group_size: usize,
    batch: usize,
    n_kv_heads: usize,
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * n_heads * seq * head_dim];
    for b in 0..batch {
        for h_kv in 0..n_kv_heads {
            let input_offset_base = ((b * n_kv_heads + h_kv) * seq) * head_dim;
            for g in 0..group_size {
                let h = h_kv * group_size + g;
                let output_offset_base = ((b * n_heads + h) * seq) * head_dim;
                for s in 0..seq {
                    let input_offset = input_offset_base + s * head_dim;
                    let output_offset = output_offset_base + s * head_dim;
                    output[output_offset..output_offset + head_dim].copy_from_slice(&input[input_offset..input_offset + head_dim]);
                }
            }
        }
    }
    output
}

#[test]
fn test_repeat_kv_heads_kernel_matches_cpu() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let batch = 2usize;
    let n_kv_heads = 2usize;
    let group_size = 3usize;
    let n_heads = n_kv_heads * group_size;
    let seq = 4usize;
    let head_dim = 5usize;

    let element_count = batch * n_kv_heads * seq * head_dim;
    let input_data: Vec<f32> = (0..element_count).map(|v| v as f32).collect();
    let input = Tensor::create_tensor_from_slice(&input_data, vec![batch * n_kv_heads, seq, head_dim], &ctx)?;

    let expected = cpu_repeat_kv_heads(&input_data, group_size, batch, n_kv_heads, n_heads, seq, head_dim);

    let output = ctx.call::<RepeatKvHeadsOp>((
        input,
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        seq as u32,
    ))?;
    ctx.synchronize();

    assert_eq!(output.dims(), &[batch * n_heads, seq, head_dim]);
    let gpu_values = output.as_slice();
    assert_eq!(gpu_values.len(), expected.len());
    for (idx, (gpu, cpu)) in gpu_values.iter().zip(expected.iter()).enumerate() {
        assert!((gpu - cpu).abs() < 1e-5, "Mismatch at index {}: gpu={} expected={}", idx, gpu, cpu);
    }

    Ok(())
}
