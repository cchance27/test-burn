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
    let cache_capacity = seq;
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
        cache_capacity as u32,
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

#[test]
fn test_incremental_repeated_cache_matches_kernel() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let batch = 2usize;
    let n_kv_heads = 2usize;
    let group_size = 3usize;
    let n_heads = n_kv_heads * group_size;
    let seq = 4usize;
    let cache_capacity = seq + 3;
    let head_dim = 5usize;

    let layer_idx = 0usize;
    ctx.alloc_kv_cache(layer_idx, cache_capacity, batch * n_kv_heads, batch * n_heads, head_dim)?;

    for step in 0..seq {
        let mut k_values = Vec::with_capacity(batch * n_kv_heads * head_dim);
        let mut v_values = Vec::with_capacity(batch * n_kv_heads * head_dim);
        for bh in 0..batch * n_kv_heads {
            for d in 0..head_dim {
                k_values.push((step * 100 + bh * 10 + d) as f32);
                v_values.push((step * 100 + bh * 10 + d) as f32 + 0.5);
            }
        }

        let k_step = Tensor::create_tensor_from_slice(&k_values, vec![batch * n_kv_heads, 1, head_dim], &ctx)?;
        let v_step = Tensor::create_tensor_from_slice(&v_values, vec![batch * n_kv_heads, 1, head_dim], &ctx)?;

        ctx.write_kv_step(layer_idx, step, &k_step, &v_step)?;
        ctx.write_repeated_kv_step(layer_idx, step, group_size, &k_step, &v_step)?;
    }

    let entry = ctx
        .kv_caches
        .get(&layer_idx)
        .cloned()
        .expect("kv cache must exist after allocation");

    let (k_canonical_view, cache_stride) = ctx.kv_cache_history_view(&entry.k, seq)?;
    let (v_canonical_view, _) = ctx.kv_cache_history_view(&entry.v, seq)?;
    let (k_repeated_view, _) = ctx.kv_cache_history_view(&entry.repeated_k, seq)?;
    let (v_repeated_view, _) = ctx.kv_cache_history_view(&entry.repeated_v, seq)?;

    let expected_k = ctx.call::<RepeatKvHeadsOp>((
        k_canonical_view.clone(),
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        cache_stride as u32,
    ))?;
    let expected_v = ctx.call::<RepeatKvHeadsOp>((
        v_canonical_view.clone(),
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        cache_stride as u32,
    ))?;

    ctx.synchronize();

    let actual_k = k_repeated_view.as_slice();
    let actual_v = v_repeated_view.as_slice();
    let expected_k_slice = expected_k.as_slice();
    let expected_v_slice = expected_v.as_slice();

    assert_eq!(actual_k.len(), expected_k_slice.len());
    assert_eq!(actual_v.len(), expected_v_slice.len());

    for (idx, (actual, expected)) in actual_k.iter().zip(expected_k_slice.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "K repeat mismatch at element {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }

    for (idx, (actual, expected)) in actual_v.iter().zip(expected_v_slice.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "V repeat mismatch at element {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }

    Ok(())
}
