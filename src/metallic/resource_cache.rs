use super::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey, MpsSoftMaxKey, SdpaKey},
    cacheable::Cacheable,
    cacheable_resources::{CacheableMpsGemm, CacheableMpsMatrixDescriptor, CacheableMpsSoftMax},
    cacheable_sdpa::CacheableSdpa,
    error::MetalError,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use rustc_hash::{FxHashMap, FxHasher};
use std::collections::hash_map::{Entry, RawEntryMut};
use std::hash::{Hash, Hasher};

/// A generic resource cache that uses FxHashMap for high-performance in-memory key-value storage.
///
/// This cache is designed to store and retrieve cacheable resources efficiently.
pub struct ResourceCache {
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    gemm_cache: FxHashMap<MpsGemmKey, CacheableMpsGemm>,
    descriptor_cache: FxHashMap<MpsMatrixDescriptorKey, CacheableMpsMatrixDescriptor>,
    softmax_cache: FxHashMap<MpsSoftMaxKey, CacheableMpsSoftMax>,
    sdpa_cache: FxHashMap<SdpaKey, CacheableSdpa>,
    permute_constant_cache: FxHashMap<PermuteConstantKey, CacheableConstantBuffer>,
    permute_constant_cache_hits: usize,
    permute_constant_cache_misses: usize,
    permute_inline_uploads: usize,
    permute_inline_bytes: usize,
}

impl ResourceCache {
    /// Create a new, empty resource cache.
    pub fn new() -> Self {
        Self::with_default_device(None)
    }

    /// Create a new resource cache that will fall back to the provided device when callers
    /// don't supply one explicitly.
    #[inline]
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
            permute_constant_cache: FxHashMap::default(),
            permute_constant_cache_hits: 0,
            permute_constant_cache_misses: 0,
            permute_inline_uploads: 0,
            permute_inline_bytes: 0,
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
    #[inline]
    pub fn get_or_create_gemm(
        &mut self,
        key: MpsGemmKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixMultiplication>, MetalError> {
        let cacheable_gemm = Self::get_or_create_resource(&mut self.gemm_cache, key, Some(device), self.default_device.as_ref())?;
        Ok(cacheable_gemm.gemm.clone())
    }

    /// Get a cached matrix descriptor by key, or create it if it doesn't exist.
    #[inline]
    pub fn get_or_create_descriptor(
        &mut self,
        key: MpsMatrixDescriptorKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixDescriptor>, MetalError> {
        let cacheable_descriptor =
            Self::get_or_create_resource(&mut self.descriptor_cache, key, Some(device), self.default_device.as_ref())?;
        Ok(cacheable_descriptor.descriptor.clone())
    }

    /// Get or create an MPS softmax operation.
    #[inline]
    pub fn get_or_create_softmax(
        &mut self,
        key: MpsSoftMaxKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixSoftMax>, MetalError> {
        let cacheable_softmax = Self::get_or_create_resource(&mut self.softmax_cache, key, Some(device), self.default_device.as_ref())?;
        Ok(cacheable_softmax.softmax.clone())
    }

    /// Get or create an SDPA operation.
    #[inline]
    pub fn get_or_create_sdpa(&mut self, batch: usize, seq_q: usize, seq_k: usize, dim: usize) -> CacheableSdpa {
        let key = SdpaKey { batch, seq_q, seq_k, dim };
        // SDPA creation should never fail, so we unwrap.
        Self::get_or_create_resource(&mut self.sdpa_cache, key, None, self.default_device.as_ref())
            .unwrap()
            .clone()
    }

    /// Get or create a reusable constant buffer for permute kernels.
    #[inline]
    pub fn get_or_create_permute_constant_buffer(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        kind: PermuteConstantKind,
        data: &[u32],
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, MetalError> {
        let length = data.len() * std::mem::size_of::<u32>();

        if length <= PERMUTE_INLINE_BYTE_LIMIT {
            self.permute_inline_uploads += 1;
            self.permute_inline_bytes += length;
            return Ok(None);
        }

        let hash = PermuteConstantKey::hash_slice(kind, data);

        match self
            .permute_constant_cache
            .raw_entry_mut()
            .from_hash(hash, |key| key.kind == kind && key.data.as_ref() == data)
        {
            RawEntryMut::Occupied(entry) => {
                self.permute_constant_cache_hits += 1;
                Ok(Some(entry.get().buffer.clone()))
            }
            RawEntryMut::Vacant(entry) => {
                let buffer = device
                    .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
                    .ok_or(MetalError::BufferCreationFailed(length))?;

                if length > 0 {
                    unsafe {
                        let dst = buffer.contents().as_ptr().cast::<u8>();
                        std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, length);
                    }
                }

                self.permute_constant_cache_misses += 1;

                let key = PermuteConstantKey::owned(kind, data);
                let buffer_clone = buffer.clone();
                entry.insert(hash, key, CacheableConstantBuffer { buffer, length });

                Ok(Some(buffer_clone))
            }
        }
    }

    /// Get statistics about the cache.
    #[inline]
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm_cache_size: self.gemm_cache.len(),
            descriptor_cache_size: self.descriptor_cache.len(),
            softmax_cache_size: self.softmax_cache.len(),
            sdpa_cache_size: self.sdpa_cache.len(),
            permute_constant_cache_size: self.permute_constant_cache.len(),
            permute_constant_cache_hits: self.permute_constant_cache_hits,
            permute_constant_cache_misses: self.permute_constant_cache_misses,
            permute_inline_uploads: self.permute_inline_uploads,
            permute_inline_bytes: self.permute_inline_bytes,
        }
    }

    /// Clears all internal caches.
    pub fn clear(&mut self) {
        self.gemm_cache.clear();
        self.descriptor_cache.clear();
        self.softmax_cache.clear();
        self.sdpa_cache.clear();
        self.permute_constant_cache.clear();
        self.permute_constant_cache_hits = 0;
        self.permute_constant_cache_misses = 0;
        self.permute_inline_uploads = 0;
        self.permute_inline_bytes = 0;
    }
}

/// Statistics about the cache.
#[derive(Debug)]
pub struct CacheStats {
    pub gemm_cache_size: usize,
    pub descriptor_cache_size: usize,
    pub softmax_cache_size: usize,
    pub sdpa_cache_size: usize,
    pub permute_constant_cache_size: usize,
    pub permute_constant_cache_hits: usize,
    pub permute_constant_cache_misses: usize,
    pub permute_inline_uploads: usize,
    pub permute_inline_bytes: usize,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub enum PermuteConstantKind {
    SrcStrides,
    DstStrides,
    Dims,
    Permutation,
}

#[derive(Hash, Eq, PartialEq)]
pub struct PermuteConstantKey {
    kind: PermuteConstantKind,
    data: Box<[u32]>,
}

impl PermuteConstantKey {
    fn owned(kind: PermuteConstantKind, data: &[u32]) -> Self {
        Self {
            kind,
            data: data.to_vec().into_boxed_slice(),
        }
    }

    fn hash_slice(kind: PermuteConstantKind, data: &[u32]) -> u64 {
        let mut hasher = FxHasher::default();
        kind.hash(&mut hasher);
        data.hash(&mut hasher);
        hasher.finish()
    }
}

pub const PERMUTE_INLINE_BYTE_LIMIT: usize = 4 * 1024;

#[derive(Clone)]
pub struct CacheableConstantBuffer {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub length: usize,
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}
