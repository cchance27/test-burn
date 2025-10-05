use super::*;

#[test]
fn test_cacheable_trait() {
    // Test CacheableSdpa since it doesn't require complex objects
    let key = SdpaKey {
        batch: 2,
        dim: 64,
        dtype: crate::metallic::tensor::dtypes::Dtype::F32,
    };
    let sdpa = CacheableSdpa::from_key(&key, None).unwrap();
    let expected_key = SdpaKey {
        batch: 2,
        dim: 64,
        dtype: crate::metallic::tensor::dtypes::Dtype::F32,
    };
    assert_eq!(sdpa.cache_key(), expected_key);
}

#[test]
fn sdpa_cache_hits_increase_for_repeated_requests() {
    use crate::metallic::resource_cache::ResourceCache;
    use crate::metallic::tensor::dtypes::Dtype;

    let mut cache = ResourceCache::new();
    let dtype = Dtype::F32;

    // Initial miss populates the cache.
    let _ = cache.get_or_create_sdpa(8, 64, dtype);
    let stats_after_miss = cache.get_stats();
    assert_eq!(stats_after_miss.sdpa.misses, 1);
    assert_eq!(stats_after_miss.sdpa.hits, 0);

    // Subsequent lookups with the same stable key should hit regardless of
    // external sequence growth.
    for _ in 0..3 {
        let _ = cache.get_or_create_sdpa(8, 64, dtype);
    }

    let stats_after_hits = cache.get_stats();
    assert_eq!(stats_after_hits.sdpa.misses, 1);
    assert_eq!(stats_after_hits.sdpa.hits, 3);
}
