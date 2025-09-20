use super::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey, SdpaKey},
    cacheable::Cacheable,
    cacheable_resources::{CacheableMpsGemm, CacheableMpsMatrixDescriptor},
    cacheable_sdpa::CacheableSdpa,
    error::MetalError,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use rustc_hash::FxHashMap;

/// A generic resource cache that uses FxHashMap for high-performance in-memory key-value storage.
///
/// This cache is designed to store and retrieve cacheable resources efficiently.
pub struct ResourceCache {
    gemm_cache: FxHashMap<MpsGemmKey, CacheableMpsGemm>,
    descriptor_cache: FxHashMap<MpsMatrixDescriptorKey, CacheableMpsMatrixDescriptor>,
    sdpa_cache: FxHashMap<SdpaKey, CacheableSdpa>,
}

impl ResourceCache {
    /// Create a new, empty resource cache.
    pub fn new() -> Self {
        Self {
            gemm_cache: FxHashMap::default(),
            descriptor_cache: FxHashMap::default(),
            sdpa_cache: FxHashMap::default(),
        }
    }

    /// Get a cached resource by key, or create it if it doesn't exist.
    fn get_or_create_resource<'a, C: Cacheable>(
        cache: &'a mut FxHashMap<C::Key, C>,
        key: C::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&'a C, MetalError>
    where
        C::Key: std::hash::Hash + Eq + Clone,
    {
        if !cache.contains_key(&key) {
            let resource = match device {
                Some(d) => C::from_key(&key, d)?,
                None => {
                    // This is for device-independent resources like SDPA
                    #[allow(clippy::transmute_ptr_to_ref)]
                    let dummy_device: &Retained<ProtocolObject<dyn MTLDevice>> =
                        unsafe { std::mem::transmute(&() as *const ()) };
                    C::from_key(&key, dummy_device)?
                }
            };
            cache.insert(key.clone(), resource);
        }
        cache.get(&key).ok_or_else(|| {
            MetalError::ResourceNotCached("Failed to retrieve resource from cache".to_string())
        })
    }

    /// Get a cached GEMM operation by key, or create it if it doesn't exist.
    pub fn get_or_create_gemm(
        &mut self,
        key: MpsGemmKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixMultiplication>, MetalError>
    {
        let cacheable_gemm = Self::get_or_create_resource(&mut self.gemm_cache, key, Some(device))?;
        Ok(cacheable_gemm.gemm.clone())
    }

    /// Get a cached matrix descriptor by key, or create it if it doesn't exist.
    pub fn get_or_create_descriptor(
        &mut self,
        key: MpsMatrixDescriptorKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixDescriptor>, MetalError> {
        let cacheable_descriptor =
            Self::get_or_create_resource(&mut self.descriptor_cache, key, Some(device))?;
        Ok(cacheable_descriptor.descriptor.clone())
    }

    /// Get or create an SDPA operation.
    pub fn get_or_create_sdpa(
        &mut self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
    ) -> CacheableSdpa {
        let key = SdpaKey {
            batch,
            seq_q,
            seq_k,
            dim,
        };
        // SDPA creation should never fail, so we unwrap.
        Self::get_or_create_resource(&mut self.sdpa_cache, key, None)
            .unwrap()
            .clone()
    }
}

/// Statistics about the cache.
#[derive(Debug)]
#[allow(dead_code)]
pub struct CacheStats {
    pub gemm_cache_size: usize,
    pub descriptor_cache_size: usize,
    pub sdpa_cache_size: usize,
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}
