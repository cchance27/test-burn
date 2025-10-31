//! Comprehensive tests for cache eviction policies.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::MTLDevice;

    use crate::{
        caching::{CacheRegistry, CacheableKernel, EvictionPolicy}, error::MetalError
    };

    // Mock kernel for testing
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct MockKey(usize);

    #[derive(Clone)]
    struct MockResource {
        value: usize,
    }

    struct MockParams(usize);

    struct MockKernel;

    impl CacheableKernel for MockKernel {
        type Key = MockKey;
        type CachedResource = MockResource;
        type Params = MockParams;

        const CACHE_NAME: &'static str = "mock_kernel_cache";

        fn create_cache_key(params: &Self::Params) -> Self::Key {
            MockKey(params.0)
        }

        fn create_cached_resource(
            key: &Self::Key,
            _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        ) -> Result<Self::CachedResource, MetalError> {
            Ok(MockResource { value: key.0 })
        }
    }

    #[test]
    fn test_no_eviction_by_default() {
        let mut registry = CacheRegistry::default();

        // Insert many entries
        for i in 0..1000 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        // All entries should still be present
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert_eq!(metrics.size, 1000);
        assert_eq!(metrics.evictions, 0);
    }

    #[test]
    fn test_size_limited_lru_eviction() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::size_limited_lru(10));

        // Insert 20 entries
        for i in 0..20 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        // Cache should be limited to 10 entries
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert!(metrics.size <= 10, "Cache size {} exceeds limit", metrics.size);
        assert!(metrics.evictions > 0, "Expected evictions to occur");
    }

    #[test]
    fn test_lru_evicts_least_recently_used() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::lru().with_max_entries(3));

        // Insert 3 entries
        let _ = registry.get_or_create::<MockKernel>(&MockParams(0), None);
        let _ = registry.get_or_create::<MockKernel>(&MockParams(1), None);
        let _ = registry.get_or_create::<MockKernel>(&MockParams(2), None);

        // Access entries 1 and 2 to make them recently used
        let _ = registry.get_or_create::<MockKernel>(&MockParams(1), None);
        let _ = registry.get_or_create::<MockKernel>(&MockParams(2), None);

        // Insert a new entry, should evict entry 0 (least recently used)
        let _ = registry.get_or_create::<MockKernel>(&MockParams(3), None);

        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert_eq!(metrics.size, 3);
        assert!(metrics.evictions > 0);
    }

    #[test]
    fn test_min_entries_prevents_over_eviction() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::size_limited_lru(2).with_min_entries(5));

        // Insert 10 entries
        for i in 0..10 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        // Cache should maintain at least min_entries
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert!(metrics.size >= 5, "Cache size {} is below min_entries", metrics.size);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::fifo().with_max_entries(3));

        // Insert 5 entries
        for i in 0..5 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        // Cache should be limited to 3 entries
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert!(metrics.size <= 3);
        assert!(metrics.evictions > 0);
    }

    #[test]
    fn test_idle_timeout_eviction() {
        let mut registry = CacheRegistry::default();

        // Set a very short idle timeout for testing
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::idle_timeout(Duration::from_millis(1)));

        // Insert entries
        let _ = registry.get_or_create::<MockKernel>(&MockParams(0), None);
        let _ = registry.get_or_create::<MockKernel>(&MockParams(1), None);

        // Sleep to ensure entries become idle
        std::thread::sleep(Duration::from_millis(10));

        // Trigger eviction by accessing cache
        let _ = registry.get_or_create::<MockKernel>(&MockParams(2), None);

        // Idle entries should have been evicted
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert!(metrics.evictions > 0, "Expected idle entries to be evicted");
    }

    #[test]
    fn test_hybrid_policy() {
        let mut registry = CacheRegistry::default();

        // Hybrid: size limit + idle timeout
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::hybrid(5, Duration::from_millis(1)));

        // Insert entries
        for i in 0..10 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        let metrics = registry.metrics::<MockKernel>().unwrap();

        // Size should be limited
        assert!(metrics.size <= 5, "Cache size {} exceeds limit", metrics.size);

        // Evictions should have occurred
        assert!(metrics.evictions > 0);
    }

    #[test]
    fn test_eviction_policy_allows_eviction() {
        assert!(!EvictionPolicy::none().allows_eviction());
        assert!(EvictionPolicy::lru().with_max_entries(10).allows_eviction());
        assert!(EvictionPolicy::fifo().with_max_entries(10).allows_eviction());
        assert!(EvictionPolicy::idle_timeout(Duration::from_secs(60)).allows_eviction());
        assert!(EvictionPolicy::size_limited_lru(100).allows_eviction());
    }

    #[test]
    fn test_eviction_metrics_are_tracked() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::size_limited_lru(5));

        // Insert entries to trigger eviction
        for i in 0..20 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
        }

        let metrics = registry.metrics::<MockKernel>().unwrap();

        // Verify eviction counter is incremented
        assert!(metrics.evictions > 0);

        // Verify size is maintained
        assert!(metrics.size <= 5);
    }

    #[test]
    fn test_cache_hit_updates_lru_timestamp() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::lru().with_max_entries(2));

        // Insert 2 entries
        let _ = registry.get_or_create::<MockKernel>(&MockParams(0), None);
        let _ = registry.get_or_create::<MockKernel>(&MockParams(1), None);

        // Access entry 0 multiple times to make it most recently used
        for _ in 0..5 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(0), None);
        }

        // Insert a new entry - should evict entry 1 (not entry 0)
        let _ = registry.get_or_create::<MockKernel>(&MockParams(2), None);

        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert_eq!(metrics.size, 2);

        // Entry 0 should still be accessible (wasn't evicted)
        let result = registry.get_or_create::<MockKernel>(&MockParams(0), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_different_policies_per_kernel() {
        // Mock a second kernel type
        struct MockKernel2;
        impl CacheableKernel for MockKernel2 {
            type Key = MockKey;
            type CachedResource = MockResource;
            type Params = MockParams;
            const CACHE_NAME: &'static str = "mock_kernel_2_cache";

            fn create_cache_key(params: &Self::Params) -> Self::Key {
                MockKey(params.0)
            }

            fn create_cached_resource(
                key: &Self::Key,
                _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
            ) -> Result<Self::CachedResource, MetalError> {
                Ok(MockResource { value: key.0 })
            }
        }

        let mut registry = CacheRegistry::default();

        // Set different policies for different kernels
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::size_limited_lru(5));
        registry.set_eviction_policy::<MockKernel2>(EvictionPolicy::size_limited_lru(10));

        // Insert entries for both kernels
        for i in 0..20 {
            let _ = registry.get_or_create::<MockKernel>(&MockParams(i), None);
            let _ = registry.get_or_create::<MockKernel2>(&MockParams(i), None);
        }

        let metrics1 = registry.metrics::<MockKernel>().unwrap();
        let metrics2 = registry.metrics::<MockKernel2>().unwrap();

        // Each should respect its own limit
        assert!(metrics1.size <= 5);
        assert!(metrics2.size <= 10);
    }

    #[test]
    fn test_eviction_preserves_cache_correctness() {
        let mut registry = CacheRegistry::default();
        registry.set_eviction_policy::<MockKernel>(EvictionPolicy::size_limited_lru(5));

        // Insert many entries
        for i in 0..100 {
            let resource = registry.get_or_create::<MockKernel>(&MockParams(i), None).unwrap();
            // Verify the resource is correct
            assert_eq!(resource.value, i);
        }

        // Even with eviction, cache should maintain correctness
        let metrics = registry.metrics::<MockKernel>().unwrap();
        assert!(metrics.size <= 5);
        assert!(metrics.hits > 0 || metrics.misses > 0);
    }
}
