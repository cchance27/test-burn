use super::{cache_keys::*, error::MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// A trait for types that can be cached in the resource cache.
/// 
/// Types that implement this trait can be stored in and retrieved from
/// the resource cache using a unique key.
pub trait Cacheable: Clone {
    /// The type of key used to identify this cacheable resource.
    type Key: Clone + std::hash::Hash + Eq;

    /// Get the cache key for this resource.
    /// 
    /// This key uniquely identifies the resource and is used to look it up
    /// in the cache.
    fn cache_key(&self) -> Self::Key;
    
    /// Create a new instance of this resource from its key and a device reference.
    /// 
    /// This method is used by the resource cache to create new instances of
    /// cached resources when they are not found in the cache.
    fn from_key(
        key: &Self::Key,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Self, MetalError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metallic::cacheable_sdpa::CacheableSdpa;
    use crate::metallic::cache_keys::SdpaKey;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLDevice;

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
}