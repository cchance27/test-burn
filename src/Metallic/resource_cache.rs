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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zerovec::{ZeroMap, ZeroVec};

/// A generic resource cache that uses zeromap for high-performance key-value storage.
/// 
/// This cache is designed to store and retrieve cacheable resources efficiently,
/// allowing for fine-grained caching of individual components rather than
/// monolithic cache structures.
/// 
/// For Metal resources, we store the keys in zeromap and recreate the resources
/// on demand since Metal resources cannot be serialized.
pub struct ResourceCache {
    // Store keys in zeromap for efficient lookup
    gemm_keys: HashMap<u64, MpsGemmKey>,
    descriptor_keys: HashMap<u64, MpsMatrixDescriptorKey>,
    sdpa_keys: HashMap<u64, SdpaKey>,
    // Store the actual resources in HashMap for now
    // In the future, we might want to implement a more sophisticated caching
    // strategy that can evict unused resources
    gemm_cache: HashMap<MpsGemmKey, CacheableMpsGemm>,
    descriptor_cache: HashMap<MpsMatrixDescriptorKey, CacheableMpsMatrixDescriptor>,
    sdpa_cache: HashMap<SdpaKey, CacheableSdpa>,
}

impl ResourceCache {
    /// Create a new, empty resource cache.
    pub fn new() -> Self {
        Self {
            gemm_keys: HashMap::new(),
            descriptor_keys: HashMap::new(),
            sdpa_keys: HashMap::new(),
            gemm_cache: HashMap::new(),
            descriptor_cache: HashMap::new(),
            sdpa_cache: HashMap::new(),
        }
    }

    /// Get a cached resource by key, or create it if it doesn't exist.
    fn get_or_create_resource<C: Cacheable>(
        cache: &mut HashMap<C::Key, C>,
        keys: &mut HashMap<u64, C::Key>,
        key: C::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        hash_fn: fn(&C::Key) -> u64,
    ) -> Result<C, MetalError> {
        // Check if we have this key in our zeromap-like structure
        let key_hash = hash_fn(&key);
        keys.entry(key_hash).or_insert_with(|| key.clone());

        if !cache.contains_key(&key) {
            let cacheable_resource = match device {
                Some(device) => C::from_key(&key, device)?,
                None => {
                    // For types that don't need a device, we can pass a dummy reference
                    // This is safe because the from_key implementation for these types
                    // doesn't actually use the device parameter
                    #[allow(clippy::transmute_ptr_to_ref)]
                    let dummy_device: &Retained<ProtocolObject<dyn MTLDevice>> = 
                        unsafe { std::mem::transmute(&() as *const ()) };
                    C::from_key(&key, dummy_device)?
                },
            };
            let cache_key = cacheable_resource.cache_key();
            cache.insert(cache_key, cacheable_resource);
        }

        cache
            .get(&key)
            .cloned()
            .ok_or_else(|| MetalError::ResourceNotCached("Resource not found in cache".to_string()))
    }

    /// Get a cached GEMM operation by key, or create it if it doesn't exist.
    pub fn get_or_create_gemm(
        &mut self,
        key: MpsGemmKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixMultiplication>, MetalError> {
        let cacheable_gemm = Self::get_or_create_resource(
            &mut self.gemm_cache,
            &mut self.gemm_keys,
            key,
            Some(device),
            Self::hash_gemm_key_static,
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
            &mut self.descriptor_keys,
            key,
            Some(device),
            Self::hash_descriptor_key_static,
        )?;
        Ok(cacheable_descriptor.descriptor.clone())
    }

    fn hash_gemm_key_static(key: &MpsGemmKey) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_descriptor_key_static(key: &MpsMatrixDescriptorKey) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_sdpa_key_static(key: &SdpaKey) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
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
        
        // For SDPA, we don't need to use the device since it doesn't create Metal resources
        Self::get_or_create_resource(
            &mut self.sdpa_cache,
            &mut self.sdpa_keys,
            key,
            None,
            Self::hash_sdpa_key_static,
        ).unwrap() // SDPA creation should never fail
    }
}

/// Statistics about the cache.
#[derive(Debug)]
#[allow(dead_code)]
pub struct CacheStats {
    pub gemm_cache_size: usize,
    pub descriptor_cache_size: usize,
    pub sdpa_cache_size: usize,
    pub gemm_key_count: usize,
    pub descriptor_key_count: usize,
    pub sdpa_key_count: usize,
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}