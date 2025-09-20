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
    // Create a dummy device for testing (not actually used)
    #[allow(clippy::transmute_ptr_to_ref)]
    let dummy_device: &Retained<ProtocolObject<dyn MTLDevice>> =
        unsafe { std::mem::transmute(&() as *const ()) };
    let sdpa = CacheableSdpa::from_key(&key, dummy_device).unwrap();
    let expected_key = SdpaKey {
        batch: 2,
        seq_q: 8,
        seq_k: 16,
        dim: 64,
    };
    assert_eq!(sdpa.cache_key(), expected_key);
}
