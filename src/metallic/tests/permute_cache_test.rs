use crate::metallic::{Context, F32Element, MetalError, Tensor, TensorInit, TensorStorage, resource_cache::PERMUTE_INLINE_BYTE_LIMIT};

fn build_high_rank_tensor(ctx: &Context<F32Element>, rank: usize) -> Result<Tensor<F32Element>, MetalError> {
    let dims = vec![1usize; rank];
    // Only a single element is required because each dimension is size one.
    let data = [0.0f32];
    Tensor::new(dims, TensorStorage::Dedicated(ctx), TensorInit::CopyFrom(&data))
}

#[test]
fn permute_large_rank_reuses_constant_buffers() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;
    let inline_elements = PERMUTE_INLINE_BYTE_LIMIT / std::mem::size_of::<u32>();
    let rank = inline_elements + 1; // Ensure the constant payload exceeds Metal's inline threshold.
    let tensor = build_high_rank_tensor(&ctx, rank)?;
    let expected_payload_bytes = rank * std::mem::size_of::<u32>();
    let permutation: Vec<usize> = (0..rank).rev().collect();

    // First dispatch populates the cache via misses.
    let _ = tensor.permute(&permutation, &mut ctx)?;
    let stats_after_first = ctx
        .get_cache_stats()
        .expect("resource cache should be initialized after first permute");
    assert_eq!(stats_after_first.permute_constant_cache_size, 4);
    assert_eq!(stats_after_first.permute_constant_cache_hits, 0);
    assert_eq!(stats_after_first.permute_constant_cache_misses, 4);
    assert_eq!(stats_after_first.permute_inline_uploads, 0);
    assert_eq!(stats_after_first.permute_inline_bytes, 0);
    assert_eq!(stats_after_first.permute_inline_max_bytes, 0);
    assert_eq!(stats_after_first.permute_cached_max_bytes, expected_payload_bytes);

    // Subsequent dispatches should reuse the cached buffers.
    for _ in 0..3 {
        let _ = tensor.permute(&permutation, &mut ctx)?;
    }

    let stats_after_reuse = ctx
        .get_cache_stats()
        .expect("resource cache should remain available for subsequent permutes");
    assert_eq!(stats_after_reuse.permute_constant_cache_size, 4);
    assert_eq!(stats_after_reuse.permute_constant_cache_misses, 4);
    assert!(
        stats_after_reuse.permute_constant_cache_hits >= 12,
        "expected at least three cache hits per constant buffer after reuse"
    );
    assert_eq!(stats_after_reuse.permute_inline_uploads, 0);
    assert_eq!(stats_after_reuse.permute_inline_bytes, 0);
    assert_eq!(stats_after_reuse.permute_inline_max_bytes, 0);
    assert_eq!(stats_after_reuse.permute_cached_max_bytes, expected_payload_bytes);

    Ok(())
}
