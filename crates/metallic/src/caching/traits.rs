use std::fmt::Debug;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;

use crate::error::MetalError;

/// Trait implemented by kernels that need to participate in the unified cache.
///
/// The trait is intentionally lightweight so it can be implemented either on
/// the kernel itself or on a thin adapter struct that proxies an existing
/// cacheable resource.  Each implementation defines how to build the cache key
/// from high level parameters and how to create the resource when a cache miss
/// occurs.
pub trait CacheableKernel: 'static {
    /// Unique key type for entries in this cache.
    type Key: Clone + Eq + std::hash::Hash + Debug + 'static;
    /// Concrete resource stored in the cache.
    type CachedResource: Clone + 'static;
    /// Parameter bag from which cache keys are derived.
    type Params;

    /// Human readable cache name used for metrics emission.
    const CACHE_NAME: &'static str;

    /// Build a cache key from operation parameters.
    fn create_cache_key(params: &Self::Params) -> Self::Key;

    /// Instantiate a cached resource from its key.
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError>;
}
