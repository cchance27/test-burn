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
    ctx.alloc_kv_cache(layer_idx, cache_capacity, batch * n_heads, head_dim)?;

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

        ctx.write_kv_step(layer_idx, step, group_size, &k_step, &v_step)?;
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

    let expected_k = ctx.call::<RepeatKvHeadsOp>((
        canonical_k_tensor.clone(),
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        seq as u32,
    ))?;
    let expected_v = ctx.call::<RepeatKvHeadsOp>((
        canonical_v_tensor.clone(),
        group_size as u32,
        batch as u32,
        n_kv_heads as u32,
        n_heads as u32,
        seq as u32,
        head_dim as u32,
        seq as u32,
    ))?;

    ctx.synchronize();

    let repeated_k_history = ctx.kv_cache_history_view(&entry.k, seq)?.0;
    let repeated_v_history = ctx.kv_cache_history_view(&entry.v, seq)?.0;

    let expected_k_slice = expected_k.as_slice();
    let expected_v_slice = expected_v.as_slice();

    let expected_dims = expected_k.dims();
    assert_eq!(expected_dims, repeated_k_history.dims());
    assert_eq!(expected_dims, repeated_v_history.dims());

    let head_stride = entry.k.strides[0];
    let seq_stride = entry.k.strides[1];
    let dim_stride = entry.k.strides[2];
    assert_eq!(head_stride, entry.v.strides[0]);
    assert_eq!(seq_stride, entry.v.strides[1]);
    let value_stride = entry.v.strides[2];

    let repeated_k_data = entry.k.as_slice();
    let repeated_v_data = entry.v.as_slice();

    let heads = expected_dims[0];
    let steps = expected_dims[1];
    let dims = expected_dims[2];

    for head in 0..heads {
        for step_idx in 0..steps {
            for dim_idx in 0..dims {
                let expected_index = (head * steps + step_idx) * dims + dim_idx;
                let k_actual_index = head * head_stride + step_idx * seq_stride + dim_idx * dim_stride;
                let v_actual_index = head * head_stride + step_idx * seq_stride + dim_idx * value_stride;

                let k_actual = repeated_k_data[k_actual_index];
                let v_actual = repeated_v_data[v_actual_index];
                let k_expected = expected_k_slice[expected_index];
                let v_expected = expected_v_slice[expected_index];

                assert!(
                    (k_actual - k_expected).abs() < 1e-5,
                    "K repeat mismatch at head {}, step {}, dim {}: got {} expected {}",
                    head,
                    step_idx,
                    dim_idx,
                    k_actual,
                    k_expected
                );
                assert!(
                    (v_actual - v_expected).abs() < 1e-5,
                    "V repeat mismatch at head {}, step {}, dim {}: got {} expected {}",
                    head,
                    step_idx,
                    dim_idx,
                    v_actual,
                    v_expected
                );
            }
        }
    }

    Ok(())
}
