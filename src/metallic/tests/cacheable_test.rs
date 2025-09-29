use super::*;

#[test]
fn test_cacheable_trait() {
    // Test CacheableSdpa since it doesn't require complex objects
    let key = SdpaKey {
        batch: 2,
        seq_q: 8,
        seq_k: 16,
        dim: 64,
    };
    let sdpa = CacheableSdpa::from_key(&key, None).unwrap();
    let expected_key = SdpaKey {
        batch: 2,
        seq_q: 8,
        seq_k: 16,
        dim: 64,
    };
    assert_eq!(sdpa.cache_key(), expected_key);
}
