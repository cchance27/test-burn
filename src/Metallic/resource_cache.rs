use super::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey, MpsSoftMaxKey, SdpaKey},
    cacheable::Cacheable,
    cacheable_resources::{CacheableMpsGemm, CacheableMpsMatrixDescriptor, CacheableMpsSoftMax},
    cacheable_sdpa::CacheableSdpa,
    error::MetalError,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

/// A generic resource cache that uses FxHashMap for high-performance in-memory key-value storage.
///
/// This cache is designed to store and retrieve cacheable resources efficiently.
pub struct ResourceCache {
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    gemm_cache: FxHashMap<MpsGemmKey, CacheableMpsGemm>,
    descriptor_cache: FxHashMap<MpsMatrixDescriptorKey, CacheableMpsMatrixDescriptor>,
    softmax_cache: FxHashMap<MpsSoftMaxKey, CacheableMpsSoftMax>,
    sdpa_cache: FxHashMap<SdpaKey, CacheableSdpa>,
}

impl ResourceCache {
    /// Create a new, empty resource cache.
    pub fn new() -> Self {
        Self::with_default_device(None)
    }

    /// Create a new resource cache that will fall back to the provided device when callers
    /// don't supply one explicitly.
    pub fn with_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self::with_default_device(Some(device))
    }

    fn with_default_device(device: Option<Retained<ProtocolObject<dyn MTLDevice>>>) -> Self {
        Self {
            default_device: device,
            gemm_cache: FxHashMap::default(),
            descriptor_cache: FxHashMap::default(),
            softmax_cache: FxHashMap::default(),
            sdpa_cache: FxHashMap::default(),
        }
    }

    /// Get a cached resource by key, or create it if it doesn't exist.
    fn get_or_create_resource<'a, C: Cacheable>(
        cache: &'a mut FxHashMap<C::Key, C>,
        key: C::Key,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&'a mut C, MetalError>
    where
        C::Key: std::hash::Hash + Eq,
    {
        let device = explicit_device.or(default_device);

        match cache.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let resource = C::from_key(entry.key(), device)?;
                Ok(entry.insert(resource))
            }
        }
    }

    /// Get a cached GEMM operation by key, or create it if it doesn't exist.
    pub fn get_or_create_gemm(
        &mut self,
        key: MpsGemmKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixMultiplication>, MetalError> {
        let cacheable_gemm = Self::get_or_create_resource(
            &mut self.gemm_cache,
            key,
            Some(device),
            self.default_device.as_ref(),
        )?;
        Ok(cacheable_gemm.gemm.clone())
    }

    /// Get a cached matrix descriptor by key, or create it if it doesn't exist.
    pub fn get_or_create_descriptor(
        &mut self,
        key: MpsMatrixDescriptorKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixDescriptor>, MetalError> {
        let cacheable_descriptor = Self::get_or_create_resource(
            &mut self.descriptor_cache,
            key,
            Some(device),
            self.default_device.as_ref(),
        )?;
        Ok(cacheable_descriptor.descriptor.clone())
    }

    /// Get or create an MPS softmax operation.
    pub fn get_or_create_softmax(
        &mut self,
        key: MpsSoftMaxKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixSoftMax>, MetalError> {
        let cacheable_softmax = Self::get_or_create_resource(
            &mut self.softmax_cache,
            key,
            Some(device),
            self.default_device.as_ref(),
        )?;
        Ok(cacheable_softmax.softmax.clone())
    }

    /// Get or create an SDPA operation.
    pub fn get_or_create_sdpa(&mut self, batch: usize, seq_q: usize, seq_k: usize, dim: usize) -> CacheableSdpa {
        let key = SdpaKey { batch, seq_q, seq_k, dim };
        // SDPA creation should never fail, so we unwrap.
        Self::get_or_create_resource(
            &mut self.sdpa_cache,
            key,
            None,
            self.default_device.as_ref(),
        )
        .unwrap()
        .clone()
    }

    /// Get statistics about the cache.
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm_cache_size: self.gemm_cache.len(),
            descriptor_cache_size: self.descriptor_cache.len(),
            softmax_cache_size: self.softmax_cache.len(),
            sdpa_cache_size: self.sdpa_cache.len(),
        }
    }

    /// Clears all internal caches.
    pub fn clear(&mut self) {
        self.gemm_cache.clear();
        self.descriptor_cache.clear();
        self.softmax_cache.clear();
        self.sdpa_cache.clear();
    }
}

/// Statistics about the cache.
#[derive(Debug)]
pub struct CacheStats {
    pub gemm_cache_size: usize,
    pub descriptor_cache_size: usize,
    pub softmax_cache_size: usize,
    pub sdpa_cache_size: usize,
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}
