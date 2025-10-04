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
use std::{collections::hash_map::Entry, fmt};

/// A generic resource cache that uses FxHashMap for high-performance in-memory key-value storage.
///
/// This cache is designed to store and retrieve cacheable resources efficiently.
pub struct ResourceCache {
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    gemm_cache: FxHashMap<MpsGemmKey, CacheableMpsGemm>,
    descriptor_cache: FxHashMap<MpsMatrixDescriptorKey, CacheableMpsMatrixDescriptor>,
    softmax_cache: FxHashMap<MpsSoftMaxKey, CacheableMpsSoftMax>,
    sdpa_cache: FxHashMap<SdpaKey, CacheableSdpa>,
    gemm_counters: CacheCounters,
    descriptor_counters: CacheCounters,
    softmax_counters: CacheCounters,
    sdpa_counters: CacheCounters,
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
            gemm_counters: CacheCounters::default(),
            descriptor_counters: CacheCounters::default(),
            softmax_counters: CacheCounters::default(),
            sdpa_counters: CacheCounters::default(),
        }
    }

    /// Get a cached resource by key, or create it if it doesn't exist.
    fn get_or_create_resource<'a, C: Cacheable>(
        cache: &'a mut FxHashMap<C::Key, C>,
        key: C::Key,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        counters: &mut CacheCounters,
        cache_name: &'static str,
    ) -> Result<&'a mut C, MetalError>
    where
        C::Key: std::hash::Hash + Eq + fmt::Debug,
    {
        let device = explicit_device.or(default_device);

        match cache.entry(key) {
            Entry::Occupied(entry) => {
                counters.record_hit(cache_name, format!("{:?}", entry.key()));
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                counters.record_miss(cache_name, format!("{:?}", entry.key()));
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
        let cacheable_gemm = Self::get_or_create_resource(
            &mut self.gemm_cache,
            key,
            Some(device),
            self.default_device.as_ref(),
            &mut self.gemm_counters,
            "gemm",
        )?;
        Ok(cacheable_gemm.gemm.clone())
    }

    /// Get a cached matrix descriptor by key, or create it if it doesn't exist.
    #[inline]
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
            &mut self.descriptor_counters,
            "descriptor",
        )?;
        Ok(cacheable_descriptor.descriptor.clone())
    }

    /// Get or create an MPS softmax operation.
    #[inline]
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
            &mut self.softmax_counters,
            "softmax",
        )?;
        Ok(cacheable_softmax.softmax.clone())
    }

    /// Get or create an SDPA operation.
    #[inline]
    pub fn get_or_create_sdpa(&mut self, batch: usize, seq_q: usize, seq_k: usize, dim: usize) -> CacheableSdpa {
        let key = SdpaKey { batch, seq_q, seq_k, dim };
        // SDPA creation should never fail, so we unwrap.
        Self::get_or_create_resource(
            &mut self.sdpa_cache,
            key,
            None,
            self.default_device.as_ref(),
            &mut self.sdpa_counters,
            "sdpa",
        )
        .unwrap()
        .clone()
    }

    /// Get statistics about the cache.
    #[inline]
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm: CacheMetrics::from_parts(self.gemm_cache.len(), &self.gemm_counters),
            descriptor: CacheMetrics::from_parts(self.descriptor_cache.len(), &self.descriptor_counters),
            softmax: CacheMetrics::from_parts(self.softmax_cache.len(), &self.softmax_counters),
            sdpa: CacheMetrics::from_parts(self.sdpa_cache.len(), &self.sdpa_counters),
        }
    }

    /// Clears all internal caches.
    pub fn clear(&mut self) {
        let gemm_size = self.gemm_cache.len();
        self.gemm_cache.clear();
        self.gemm_counters.record_clear("gemm", &format!("cleared {gemm_size} entries"));

        let descriptor_size = self.descriptor_cache.len();
        self.descriptor_cache.clear();
        self.descriptor_counters
            .record_clear("descriptor", &format!("cleared {descriptor_size} entries"));

        let softmax_size = self.softmax_cache.len();
        self.softmax_cache.clear();
        self.softmax_counters
            .record_clear("softmax", &format!("cleared {softmax_size} entries"));

        let sdpa_size = self.sdpa_cache.len();
        self.sdpa_cache.clear();
        self.sdpa_counters.record_clear("sdpa", &format!("cleared {sdpa_size} entries"));
    }
}

#[derive(Clone, Debug, Default)]
struct CacheCounters {
    hits: u64,
    misses: u64,
    last_event: Option<CacheEvent>,
}

impl CacheCounters {
    fn record_hit(&mut self, cache: &'static str, detail: String) {
        self.hits += 1;
        self.last_event = Some(CacheEvent::new(CacheEventKind::Hit, cache, detail));
    }

    fn record_miss(&mut self, cache: &'static str, detail: String) {
        self.misses += 1;
        self.last_event = Some(CacheEvent::new(CacheEventKind::MissCreate, cache, detail));
    }

    fn record_clear(&mut self, cache: &'static str, reason: &str) {
        self.last_event = Some(CacheEvent::new(CacheEventKind::Cleared, cache, reason.to_string()));
    }
}

/// Statistics about an individual cache.
#[derive(Clone, Debug)]
pub struct CacheMetrics {
    pub size: usize,
    pub hits: u64,
    pub misses: u64,
    pub last_event: Option<CacheEvent>,
}

impl CacheMetrics {
    fn from_parts(size: usize, counters: &CacheCounters) -> Self {
        Self {
            size,
            hits: counters.hits,
            misses: counters.misses,
            last_event: counters.last_event.clone(),
        }
    }
}

/// Statistics about the cache.
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub gemm: CacheMetrics,
    pub descriptor: CacheMetrics,
    pub softmax: CacheMetrics,
    pub sdpa: CacheMetrics,
}

/// The type of cache event that most recently occurred for a cache.
#[derive(Clone, Copy, Debug)]
pub enum CacheEventKind {
    Hit,
    MissCreate,
    Cleared,
}

impl fmt::Display for CacheEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hit => f.write_str("hit"),
            Self::MissCreate => f.write_str("miss-create"),
            Self::Cleared => f.write_str("cleared"),
        }
    }
}

/// Summary of the most recent cache interaction.
#[derive(Clone, Debug)]
pub struct CacheEvent {
    pub kind: CacheEventKind,
    pub cache: &'static str,
    pub detail: String,
}

impl CacheEvent {
    fn new(kind: CacheEventKind, cache: &'static str, detail: String) -> Self {
        Self { kind, cache, detail }
    }
}

impl fmt::Display for CacheEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} -> {}", self.cache, self.kind, self.detail)
    }
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}
