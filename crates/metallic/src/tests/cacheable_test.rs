use super::*;

#[test]
fn test_cacheable_trait() {
    use crate::cache_keys::SeqKBucket;

    // Test CacheableSdpa since it doesn't require complex objects
    let key = SdpaKey {
        batch: 2,
        dim: 64,
        dtype: crate::tensor::dtypes::Dtype::F32,
        causal: false,
        seq_k_bucket: SeqKBucket::from(64),
        transpose_k: false,
    };
    let sdpa = CacheableSdpa::from_key(&key, None).unwrap();
    let expected_key = SdpaKey {
        batch: 2,
        dim: 64,
        dtype: crate::tensor::dtypes::Dtype::F32,
        causal: false,
        seq_k_bucket: SeqKBucket::from(64),
        transpose_k: false,
    };
    assert_eq!(sdpa.cache_key(), expected_key);
}

#[test]
fn sdpa_cache_hits_increase_for_repeated_requests() {
    use crate::{resource_cache::ResourceCache, tensor::dtypes::Dtype};

    let mut cache = ResourceCache::new();
    let dtype = Dtype::F32;

    // Initial miss populates the cache.
    let _ = cache.get_or_create_sdpa_full(8, 64, dtype, false, 64, false);
    let stats_after_miss = cache.get_stats();
    assert_eq!(stats_after_miss.sdpa.misses, 1);
    assert_eq!(stats_after_miss.sdpa.hits, 0);

    // Subsequent lookups with the same stable key should hit regardless of
    // external sequence growth.
    for _ in 0..3 {
        let _ = cache.get_or_create_sdpa_full(8, 64, dtype, false, 64, false);
    }

    let stats_after_hits = cache.get_stats();
    assert_eq!(stats_after_hits.sdpa.misses, 1);
    assert_eq!(stats_after_hits.sdpa.hits, 3);
}

#[test]
fn sdpa_incremental_decode_hits_cache_and_matches_full_attention() -> Result<(), MetalError> {
    use crate::tensor::{F32Element, TensorStorage};

    let mut ctx = Context::<F32Element>::new()?;
    let batch = 1;
    let dim = 4;
    let prefill = 3;
    let total = 4;

    let q_data: Vec<f32> = (0..batch * total * dim).map(|i| (i as f32) * 0.01).collect();
    let k_data: Vec<f32> = (0..batch * total * dim).map(|i| (i as f32) * 0.02).collect();
    let v_data: Vec<f32> = (0..batch * total * dim).map(|i| (i as f32) * 0.03).collect();

    let q_full = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx), &q_data)?;
    let k_full = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx), &k_data)?;
    let v_full = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx), &v_data)?;

    let q_prefill = q_full
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;
    let k_prefill = k_full
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;
    let v_prefill = v_full
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;

    let _ = ctx.scaled_dot_product_attention_with_offset(&q_prefill, &k_prefill, &v_prefill, true, 0)?;
    ctx.synchronize();

    let stats_after_prefill = ctx.get_cache_stats().expect("cache stats should be available after prefill");

    let decode_out = ctx.scaled_dot_product_attention_with_offset(&q_full, &k_full, &v_full, true, prefill)?;
    ctx.synchronize();

    let stats_after_decode = ctx.get_cache_stats().expect("cache stats should be available after decode");

    assert_eq!(stats_after_decode.sdpa.misses, stats_after_prefill.sdpa.misses);
    assert!(stats_after_decode.sdpa.hits >= stats_after_prefill.sdpa.hits);

    let mut baseline_ctx = Context::<F32Element>::new()?;
    let q_full_baseline = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut baseline_ctx), &q_data)?;
    let k_full_baseline = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut baseline_ctx), &k_data)?;
    let v_full_baseline = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut baseline_ctx), &v_data)?;

    let full_out = baseline_ctx.scaled_dot_product_attention_with_offset(&q_full_baseline, &k_full_baseline, &v_full_baseline, true, 0)?;
    baseline_ctx.synchronize();

    let decode_slice = decode_out.as_slice();
    let full_slice = full_out.as_slice();
    assert_eq!(decode_slice.len(), dim);
    let start = (total - 1) * dim;
    let end = start + dim;
    for (i, (a, b)) in full_slice[start..end].iter().zip(decode_slice.iter()).enumerate() {
        if (a - b).abs() > 1e-6 {
            panic!("Mismatch at index {} in decode_slice: full={}, decode={}", i, a, b);
        }
    }

    let mut ctx_zero_offset = Context::<F32Element>::new()?;
    let q_full_zero = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx_zero_offset), &q_data)?;
    let k_full_zero = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx_zero_offset), &k_data)?;
    let v_full_zero = Tensor::<F32Element>::from_f32_slice(vec![batch, total, dim], TensorStorage::Pooled(&mut ctx_zero_offset), &v_data)?;

    let q_prefill_zero = q_full_zero
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;
    let k_prefill_zero = k_full_zero
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;
    let v_prefill_zero = v_full_zero
        .reshape(vec![batch * total, dim])?
        .slice(0..batch * prefill)?
        .reshape(vec![batch, prefill, dim])?;

    let _ = ctx_zero_offset.scaled_dot_product_attention_with_offset(&q_prefill_zero, &k_prefill_zero, &v_prefill_zero, true, 0)?;
    ctx_zero_offset.synchronize();

    let incremental_zero = ctx_zero_offset.scaled_dot_product_attention_with_offset(&q_full_zero, &k_full_zero, &v_full_zero, true, 0)?;
    ctx_zero_offset.synchronize();

    let incremental_zero_slice = incremental_zero.as_slice();
    assert_eq!(incremental_zero_slice.len(), dim);
    for (i, (a, b)) in full_slice[start..end].iter().zip(incremental_zero_slice.iter()).enumerate() {
        if (a - b).abs() > 1e-6 {
            panic!("Mismatch at index {} in incremental_zero_slice: full={}, decode={}", i, a, b);
        }
    }

    Ok(())
}
