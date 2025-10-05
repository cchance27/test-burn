use crate::metallic::{F32Element, TensorInit, TensorStorage};

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
    let mut ctx = Context::<F32Element>::new()?;

    let batch = 2usize;
    let n_kv_heads = 2usize;
    let group_size = 3usize;
    let n_heads = n_kv_heads * group_size;
    let seq = 4usize;
    let cache_capacity = seq;
    let head_dim = 5usize;

    let element_count = batch * n_kv_heads * seq * head_dim;
    let input_data: Vec<f32> = (0..element_count).map(|v| v as f32).collect();
    let input = Tensor::new(
        vec![batch * n_kv_heads, seq, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&input_data),
    )?;

    let expected = cpu_repeat_kv_heads(&input_data, group_size, batch, n_kv_heads, n_heads, seq, head_dim);

    let output = ctx.call::<RepeatKvHeadsOp>((
        input,
        None,
        0,
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
    let mut ctx = Context::<F32Element>::new()?;

    let batch = 2usize;
    let n_kv_heads = 2usize;
    let group_size = 3usize;
    let n_heads = n_kv_heads * group_size;
    let seq = 4usize;
    let cache_capacity = seq + 3;
    let head_dim = 5usize;

    let layer_idx = 0usize;
    ctx.alloc_kv_cache(layer_idx, cache_capacity, batch * n_kv_heads, head_dim)?;

    let mut canonical_k = vec![0f32; batch * n_kv_heads * seq * head_dim];
    let mut canonical_v = vec![0f32; batch * n_kv_heads * seq * head_dim];

    for step in 0..seq {
        let mut k_values = Vec::with_capacity(batch * n_kv_heads * head_dim);
        let mut v_values = Vec::with_capacity(batch * n_kv_heads * head_dim);
        for bh in 0..batch * n_kv_heads {
            for d in 0..head_dim {
                let value = (step * 100 + bh * 10 + d) as f32;
                k_values.push(value);
                v_values.push(value + 0.5);
                let base = (bh * seq + step) * head_dim + d;
                canonical_k[base] = value;
                canonical_v[base] = value + 0.5;
            }
        }

        let k_step = Tensor::new(
            vec![batch * n_kv_heads, 1, head_dim],
            TensorStorage::Dedicated(&ctx),
            TensorInit::CopyFrom(&k_values),
        )?;
        let v_step = Tensor::new(
            vec![batch * n_kv_heads, 1, head_dim],
            TensorStorage::Dedicated(&ctx),
            TensorInit::CopyFrom(&v_values),
        )?;

        ctx.write_kv_step(layer_idx, step, &k_step, &v_step)?;
    }

    let entry = ctx
        .kv_caches
        .get(&layer_idx)
        .cloned()
        .expect("kv cache must exist after allocation");

    let canonical_k_tensor = Tensor::new(
        vec![batch * n_kv_heads, seq, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&canonical_k),
    )?;
    let canonical_v_tensor = Tensor::new(
        vec![batch * n_kv_heads, seq, head_dim],
        TensorStorage::Dedicated(&ctx),
        TensorInit::CopyFrom(&canonical_v),
    )?;

    let canonical_k_history = ctx.kv_cache_history_view(&entry.k, seq)?.0;
    let canonical_v_history = ctx.kv_cache_history_view(&entry.v, seq)?.0;

    assert_eq!(canonical_k_history.dims(), &[batch * n_kv_heads, seq, head_dim]);
    assert_eq!(canonical_v_history.dims(), &[batch * n_kv_heads, seq, head_dim]);

    let canonical_k_view = ctx.materialize_contiguous_view(canonical_k_history.clone())?;
    let canonical_v_view = ctx.materialize_contiguous_view(canonical_v_history.clone())?;
    ctx.synchronize();

    let canonical_k_slice = canonical_k_view.as_slice();
    let canonical_v_slice = canonical_v_view.as_slice();
    assert_eq!(canonical_k_slice, canonical_k_tensor.as_slice());
    assert_eq!(canonical_v_slice, canonical_v_tensor.as_slice());

    let repeated_k = ctx.call::<RepeatKvHeadsOp>((
        canonical_k_history.clone(),
        None,
        0,
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        cache_capacity as u32,
    ))?;
    let repeated_v = ctx.call::<RepeatKvHeadsOp>((
        canonical_v_history.clone(),
        None,
        0,
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        cache_capacity as u32,
    ))?;

    ctx.synchronize();

    let repeated_k_slice = repeated_k.as_slice();
    let repeated_v_slice = repeated_v.as_slice();
    let expected_dims = repeated_k.dims();
    assert_eq!(expected_dims, &[batch * n_heads, seq, head_dim]);

    let cpu_expected_k = cpu_repeat_kv_heads(canonical_k_slice, group_size, batch, n_kv_heads, n_heads, seq, head_dim);
    let cpu_expected_v = cpu_repeat_kv_heads(canonical_v_slice, group_size, batch, n_kv_heads, n_heads, seq, head_dim);

    assert_eq!(repeated_k_slice, cpu_expected_k.as_slice());
    assert_eq!(repeated_v_slice, cpu_expected_v.as_slice());

    // Incrementally fill a reusable buffer and ensure only new regions are written.
    let mut repeated_workspace = Tensor::zeros(vec![batch * n_heads, cache_capacity, head_dim], &mut ctx, true)?;

    // First, materialize the initial two steps.
    let delta_first = 2usize;
    let mut k_prefix = canonical_k_history.clone();
    k_prefix.dims = vec![batch * n_kv_heads, delta_first, head_dim];

    let _ = ctx.call::<RepeatKvHeadsOp>((
        k_prefix,
        Some(repeated_workspace.clone()),
        0,
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        delta_first as u32,
        head_dim as u32,
        cache_capacity as u32,
    ))?;
    // Then, write the remaining suffix without touching the prefix.
    let delta_second = seq - delta_first;
    let seq_stride = canonical_k_history.strides[1];
    let elem_size = canonical_k_history.dtype.size_bytes();
    let offset_adjust = delta_first * seq_stride * elem_size;

    let mut k_suffix = canonical_k_history.clone();
    k_suffix.offset += offset_adjust;
    k_suffix.dims = vec![batch * n_kv_heads, delta_second, head_dim];

    let _ = ctx.call::<RepeatKvHeadsOp>((
        k_suffix,
        Some(repeated_workspace.clone()),
        delta_first as u32,
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        delta_second as u32,
        head_dim as u32,
        cache_capacity as u32,
    ))?;

    ctx.synchronize();

    let mut repeated_view = repeated_workspace.clone();
    repeated_view.dims = vec![batch * n_heads, seq, head_dim];
    let repeated_contiguous = ctx.materialize_contiguous_view(repeated_view)?;
    ctx.synchronize();
    assert_eq!(repeated_contiguous.as_slice(), cpu_expected_k.as_slice());

    Ok(())
}
