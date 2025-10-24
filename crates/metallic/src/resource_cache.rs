use std::{
    collections::hash_map::Entry, fmt, time::{Duration, Instant}
};

use metallic_instrumentation::{MetricEvent, prelude::info_span, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use rustc_hash::FxHashMap;

use super::{
    cacheable::Cacheable, cacheable_resources::{
        CacheableMpsGraphFused, CacheableMpsGraphKvWrite, CacheableMpsGraphSdpa, CacheableMpsGraphSdpaMask, CacheableMpsSoftMax
    }, cacheable_sdpa::CacheableSdpa, error::MetalError
};
use crate::{
    cache_keys::{
        MpsGemmKey, MpsGraphFusedKey, MpsGraphKvWriteKey, MpsGraphSdpaKey, MpsGraphSdpaMaskKey, MpsMatrixDescriptorKey, MpsSoftMaxKey, SdpaKey, SeqKBucket
    }, cacheable_resources::{CacheableMpsGemm, CacheableMpsMatrixDescriptor}, tensor::dtypes::Dtype
};
#[derive(Clone, Debug)]
struct EntryMetadata {
    created_at: Instant,
    last_used_at: Instant,
    reuse_count: u64,
}

impl EntryMetadata {
    fn new(now: Instant) -> Self {
        Self {
            created_at: now,
            last_used_at: now,
            reuse_count: 0,
        }
    }

    fn touch(&mut self, now: Instant) {
        self.last_used_at = now;
        self.reuse_count = self.reuse_count.saturating_add(1);
    }
}

#[derive(Clone, Debug)]
struct CacheEntry<V> {
    value: V,
    metadata: EntryMetadata,
}

impl<V> CacheEntry<V> {
    fn new(value: V, now: Instant) -> Self {
        Self {
            value,
            metadata: EntryMetadata::new(now),
        }
    }

    fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

#[derive(Clone, Debug, Default)]
struct CacheLifetimeSummary {
    oldest_entry_age: Option<Duration>,
    newest_entry_age: Option<Duration>,
    longest_idle: Option<Duration>,
    shortest_idle: Option<Duration>,
    max_reuse_count: Option<u64>,
}

impl CacheLifetimeSummary {
    fn observe(&mut self, metadata: &EntryMetadata, now: Instant) {
        let age = now.saturating_duration_since(metadata.created_at);
        let idle = now.saturating_duration_since(metadata.last_used_at);

        self.oldest_entry_age = Some(self.oldest_entry_age.map_or(age, |current| current.max(age)));
        self.newest_entry_age = Some(self.newest_entry_age.map_or(age, |current| current.min(age)));
        self.longest_idle = Some(self.longest_idle.map_or(idle, |current| current.max(idle)));
        self.shortest_idle = Some(self.shortest_idle.map_or(idle, |current| current.min(idle)));
        self.max_reuse_count = Some(
            self.max_reuse_count
                .map_or(metadata.reuse_count, |current| current.max(metadata.reuse_count)),
        );
    }
}

fn summarise_lifetimes<'a, V: 'a>(entries: impl Iterator<Item = &'a CacheEntry<V>>) -> CacheLifetimeSummary {
    let now = Instant::now();
    let mut summary = CacheLifetimeSummary::default();
    for entry in entries {
        summary.observe(&entry.metadata, now);
    }
    summary
}

fn metrics_for_entries<K, V>(entries: &FxHashMap<K, CacheEntry<V>>, counters: &CacheCounters) -> CacheMetrics
where
    K: std::hash::Hash + Eq,
{
    let lifetime = summarise_lifetimes(entries.values());
    CacheMetrics::from_parts(entries.len(), counters, lifetime)
}

/// A generic resource cache that uses FxHashMap for high-performance in-memory key-value storage.
///
/// This cache is designed to store and retrieve cacheable resources efficiently.
pub struct ResourceCache {
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    gemm_cache: FxHashMap<MpsGemmKey, CacheEntry<CacheableMpsGemm>>,
    descriptor_cache: FxHashMap<MpsMatrixDescriptorKey, CacheEntry<CacheableMpsMatrixDescriptor>>,
    softmax_cache: FxHashMap<MpsSoftMaxKey, CacheEntry<CacheableMpsSoftMax>>,
    sdpa_cache: FxHashMap<SdpaKey, CacheEntry<CacheableSdpa>>,
    mpsgraph_sdpa_cache: GraphExecutableCache<MpsGraphSdpaKey, CacheableMpsGraphSdpa>,
    mpsgraph_mask_arena: MaskArena<MpsGraphSdpaMaskKey, CacheableMpsGraphSdpaMask>,
    mpsgraph_fused_cache: GraphExecutableCache<MpsGraphFusedKey, CacheableMpsGraphFused>,
    mpsgraph_kv_write_cache: GraphExecutableCache<MpsGraphKvWriteKey, CacheableMpsGraphKvWrite>,
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
            mpsgraph_sdpa_cache: GraphExecutableCache::new("mpsgraph_sdpa"),
            mpsgraph_mask_arena: MaskArena::new("mpsgraph_mask"),
            mpsgraph_fused_cache: GraphExecutableCache::new("mpsgraph_fused"),
            mpsgraph_kv_write_cache: GraphExecutableCache::new("mpsgraph_kv_write"),
            gemm_counters: CacheCounters::default(),
            descriptor_counters: CacheCounters::default(),
            softmax_counters: CacheCounters::default(),
            sdpa_counters: CacheCounters::default(),
        }
    }

    /// Get a cached resource by key, or create it if it doesn't exist.
    #[inline]
    fn get_or_create_resource<'a, C: Cacheable>(
        cache: &'a mut FxHashMap<C::Key, CacheEntry<C>>,
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

        let span = info_span!("cache_get_or_create", cache = cache_name);
        let _enter = span.enter();

        // Capture cache size before the match to avoid borrow issues
        let cache_size_before_insert = cache.len();
        let now = Instant::now();

        match cache.entry(key) {
            Entry::Occupied(entry) => {
                let detail = format!("{:?}", entry.key());
                counters.record_hit(cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", cache_name, detail),
                    hit: true,
                    bytes: 0
                }); // cache_key uses cache_name
                // Periodic summary for dashboard (every 100 ops) - for all cache types
                let ops = counters.hits.saturating_add(counters.misses);
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (counters.hits as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: cache_name.to_string(),
                        hits: counters.hits,
                        misses: counters.misses,
                        hit_rate,
                        size: cache_size_before_insert as u64, // Use captured size
                    });
                }
                let entry_ref = entry.into_mut();
                entry_ref.metadata.touch(now);
                Ok(entry_ref.value_mut())
            }
            Entry::Vacant(entry) => {
                let detail = format!("{:?}", entry.key());
                counters.record_miss(cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", cache_name, detail),
                    hit: false,
                    bytes: 0
                });
                let resource = C::from_key(entry.key(), device)?;
                let entry_ref = entry.insert(CacheEntry::new(resource, now));
                // Periodic summary for dashboard (every 100 ops) - for all cache types
                let ops = counters.hits.saturating_add(counters.misses);
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (counters.hits as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: cache_name.to_string(),
                        hits: counters.hits,
                        misses: counters.misses,
                        hit_rate,
                        size: cache_size_before_insert as u64, // Use captured size
                    });
                }
                Ok(entry_ref.value_mut())
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

    /// Get or create an MPS softmax operation with individual parameters.
    #[inline]
    pub fn get_or_create_softmax(
        &mut self,
        rows: usize,
        columns: usize,
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixSoftMax>, MetalError> {
        // For backward compatibility, use default causal value
        let seq_k_bucket = SeqKBucket::from(columns); // columns is typically seq_k in softmax
        let key = MpsSoftMaxKey {
            rows,
            columns,
            seq_k_bucket,
            causal: false, // Default value for backward compatibility
            dtype,
        };
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

    /// Get or create an MPS softmax operation with all parameters including causal flag.
    #[inline]
    pub fn get_or_create_softmax_full(
        &mut self,
        rows: usize,
        columns: usize,
        dtype: Dtype,
        causal: bool,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<objc2_metal_performance_shaders::MPSMatrixSoftMax>, MetalError> {
        let seq_k_bucket = SeqKBucket::from(columns); // columns is typically seq_k in softmax
        let key = MpsSoftMaxKey {
            rows,
            columns,
            seq_k_bucket,
            causal,
            dtype,
        };
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
    pub fn get_or_create_sdpa(&mut self, batch: usize, dim: usize, dtype: Dtype) -> CacheableSdpa {
        // For backward compatibility, create with default values for new fields
        // This is called when cache.as_mut() is Some() in SDPA
        let key = SdpaKey {
            batch,
            dim,
            dtype,
            causal: false,                   // Default value - will be overridden when needed
            seq_k_bucket: SeqKBucket::Other, // Default value - will be overridden when needed
            transpose_k: false,              // Default value - will be overridden when needed
        };
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

    /// Get or create an SDPA operation with all specialization parameters.
    #[inline]
    pub fn get_or_create_sdpa_full(
        &mut self,
        batch: usize,
        dim: usize,
        dtype: Dtype,
        causal: bool,
        seq_k: usize,
        transpose_k: bool,
    ) -> CacheableSdpa {
        let seq_k_bucket = SeqKBucket::from(seq_k);
        let key = SdpaKey {
            batch,
            dim,
            dtype,
            causal,
            seq_k_bucket,
            transpose_k,
        };
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

    /// Get or create an MPSGraph SDPA operation.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_sdpa(
        &mut self,
        batch: usize,
        _seq_q: usize,
        _seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<&mut CacheableMpsGraphSdpa, MetalError> {
        let key = MpsGraphSdpaKey {
            batch,
            dim,
            causal,
            dtype,
            accumulator_dtype,
        };
        self.mpsgraph_sdpa_cache.get_or_create(key, None, self.default_device.as_ref())
    }

    /// Get or create an MPSGraph mask buffer for SDPA with bucketed sequence lengths.
    #[inline]
    pub fn get_or_create_mpsgraph_sdpa_mask(
        &mut self,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
    ) -> Result<&mut CacheableMpsGraphSdpaMask, MetalError> {
        let key = MpsGraphSdpaMaskKey {
            causal,
            dtype,
            head_dim: dim,
            seq_q_bucket: crate::cache_keys::MaskSizeBucket::from(seq_q.max(seq_k)),
            seq_k_bucket: crate::cache_keys::MaskSizeBucket::from(seq_k),
        };
        self.mpsgraph_mask_arena.get_or_create(key, None, self.default_device.as_ref())
    }

    /// Get or create both MPSGraph SDPA executable and mask buffer in one call to avoid double borrowing.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_sdpa_and_mask(
        &mut self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<(CacheableMpsGraphSdpa, Option<CacheableMpsGraphSdpaMask>), MetalError> {
        // Get the cached graph
        let cached_graph = self.get_or_create_mpsgraph_sdpa(batch, seq_q, seq_k, dim, causal, dtype, accumulator_dtype)?;
        let graph_clone = (*cached_graph).clone();

        // Get or create the mask if needed
        let cached_mask = if causal {
            let mask = self.get_or_create_mpsgraph_sdpa_mask(seq_q, seq_k, dim, causal, dtype)?;
            Some((*mask).clone())
        } else {
            None
        };

        Ok((graph_clone, cached_mask))
    }

    /// Get or create an MPSGraph fused operation.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_fused(
        &mut self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        output_dim: usize,
        causal: bool,
        operation_type: crate::cache_keys::FusedOperationType,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<&mut CacheableMpsGraphFused, MetalError> {
        let key = crate::cache_keys::MpsGraphFusedKey {
            batch,
            seq_q,
            seq_k,
            dim,
            output_dim,
            causal,
            operation_type,
            dtype,
            accumulator_dtype,
        };
        self.mpsgraph_fused_cache.get_or_create(key, None, self.default_device.as_ref())
    }

    #[inline]
    pub fn get_or_create_mpsgraph_kv_write(
        &mut self,
        heads: usize,
        seq_bucket: usize,
        head_dim: usize,
        dtype: Dtype,
    ) -> Result<&mut CacheableMpsGraphKvWrite, MetalError> {
        let key = MpsGraphKvWriteKey {
            heads,
            seq_bucket,
            head_dim,
            dtype,
        };
        self.mpsgraph_kv_write_cache.get_or_create(key, None, self.default_device.as_ref())
    }

    /// Get statistics about the cache.
    #[inline]
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm: metrics_for_entries(&self.gemm_cache, &self.gemm_counters),
            descriptor: metrics_for_entries(&self.descriptor_cache, &self.descriptor_counters),
            softmax: metrics_for_entries(&self.softmax_cache, &self.softmax_counters),
            sdpa: metrics_for_entries(&self.sdpa_cache, &self.sdpa_counters),
            mpsgraph_sdpa: self.mpsgraph_sdpa_cache.metrics(),
            mpsgraph_mask: self.mpsgraph_mask_arena.metrics(),
            mpsgraph_fused: self.mpsgraph_fused_cache.metrics(),
            mpsgraph_kv_write: self.mpsgraph_kv_write_cache.metrics(),
        }
    }

    /// Clears all internal caches.
    pub fn clear(&mut self) {
        let gemm_size = self.gemm_cache.len();
        self.gemm_cache.clear();
        self.gemm_counters
            .record_clear("gemm", &format!("cleared {gemm_size} entries"), gemm_size as u64);

        let descriptor_size = self.descriptor_cache.len();
        self.descriptor_cache.clear();
        self.descriptor_counters
            .record_clear("descriptor", &format!("cleared {descriptor_size} entries"), descriptor_size as u64);

        let softmax_size = self.softmax_cache.len();
        self.softmax_cache.clear();
        self.softmax_counters
            .record_clear("softmax", &format!("cleared {softmax_size} entries"), softmax_size as u64);

        let sdpa_size = self.sdpa_cache.len();
        self.sdpa_cache.clear();
        self.sdpa_counters
            .record_clear("sdpa", &format!("cleared {sdpa_size} entries"), sdpa_size as u64);

        self.mpsgraph_sdpa_cache.clear();
        self.mpsgraph_mask_arena.clear();
        self.mpsgraph_fused_cache.clear();
        self.mpsgraph_kv_write_cache.clear();
    }
}

#[derive(Clone, Debug, Default)]
struct CacheCounters {
    hits: u64,
    misses: u64,
    evictions: u64,
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

    fn record_clear(&mut self, cache: &'static str, reason: &str, evicted: u64) {
        self.evictions = self.evictions.saturating_add(evicted);
        self.last_event = Some(CacheEvent::new(CacheEventKind::Cleared, cache, reason.to_string()));
    }
}

/// Statistics about an individual cache.
#[derive(Clone, Debug)]
pub struct CacheMetrics {
    pub size: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub last_event: Option<CacheEvent>,
    pub oldest_entry_age: Option<Duration>,
    pub newest_entry_age: Option<Duration>,
    pub longest_idle_age: Option<Duration>,
    pub shortest_idle_age: Option<Duration>,
    pub max_entry_reuse_count: Option<u64>,
}

impl CacheMetrics {
    fn from_parts(size: usize, counters: &CacheCounters, lifetime: CacheLifetimeSummary) -> Self {
        Self {
            size,
            hits: counters.hits,
            misses: counters.misses,
            evictions: counters.evictions,
            last_event: counters.last_event.clone(),
            oldest_entry_age: lifetime.oldest_entry_age,
            newest_entry_age: lifetime.newest_entry_age,
            longest_idle_age: lifetime.longest_idle,
            shortest_idle_age: lifetime.shortest_idle,
            max_entry_reuse_count: lifetime.max_reuse_count,
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
    pub mpsgraph_sdpa: CacheMetrics,
    pub mpsgraph_mask: CacheMetrics,
    pub mpsgraph_fused: CacheMetrics,
    pub mpsgraph_kv_write: CacheMetrics,
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

struct GraphResourceCache<K, V>
where
    V: Cacheable<Key = K>,
{
    entries: FxHashMap<K, CacheEntry<V>>,
    counters: CacheCounters,
    cache_name: &'static str,
}

impl<K, V> GraphResourceCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + fmt::Debug,
    V: Cacheable<Key = K>,
{
    fn new(cache_name: &'static str) -> Self {
        Self {
            entries: FxHashMap::default(),
            counters: CacheCounters::default(),
            cache_name,
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn metrics(&self) -> CacheMetrics {
        let lifetime = summarise_lifetimes(self.entries.values());
        CacheMetrics::from_parts(self.entries.len(), &self.counters, lifetime)
    }

    fn get_or_create(
        &mut self,
        key: K,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut V, MetalError> {
        let device = explicit_device.or(default_device);

        let span = info_span!("cache_get_or_create", cache = self.cache_name);
        let _enter = span.enter();

        let cache_size_before_insert = self.entries.len();
        let now = Instant::now();

        match self.entries.entry(key) {
            Entry::Occupied(entry) => {
                let detail = format!("{:?}", entry.key());
                self.counters.record_hit(self.cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", self.cache_name, detail),
                    hit: true,
                    bytes: 0,
                });
                let ops = self.counters.hits.saturating_add(self.counters.misses);
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (self.counters.hits as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: self.cache_name.to_string(),
                        hits: self.counters.hits,
                        misses: self.counters.misses,
                        hit_rate,
                        size: cache_size_before_insert as u64,
                    });
                }
                let entry_ref = entry.into_mut();
                entry_ref.metadata.touch(now);
                Ok(entry_ref.value_mut())
            }
            Entry::Vacant(entry) => {
                let detail = format!("{:?}", entry.key());
                self.counters.record_miss(self.cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", self.cache_name, detail),
                    hit: false,
                    bytes: 0,
                });
                let resource = V::from_key(entry.key(), device)?;
                let entry_ref = entry.insert(CacheEntry::new(resource, now));
                let ops = self.counters.hits.saturating_add(self.counters.misses);
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (self.counters.hits as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: self.cache_name.to_string(),
                        hits: self.counters.hits,
                        misses: self.counters.misses,
                        hit_rate,
                        size: cache_size_before_insert as u64,
                    });
                }
                Ok(entry_ref.value_mut())
            }
        }
    }

    fn clear(&mut self) -> usize {
        let size = self.entries.len();
        self.entries.clear();
        self.counters
            .record_clear(self.cache_name, &format!("cleared {size} entries"), size as u64);
        size
    }
}

struct GraphExecutableCache<K, V>
where
    V: Cacheable<Key = K>,
{
    inner: GraphResourceCache<K, V>,
}

impl<K, V> GraphExecutableCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + fmt::Debug,
    V: Cacheable<Key = K>,
{
    fn new(cache_name: &'static str) -> Self {
        Self {
            inner: GraphResourceCache::new(cache_name),
        }
    }

    fn get_or_create(
        &mut self,
        key: K,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut V, MetalError> {
        self.inner.get_or_create(key, explicit_device, default_device)
    }

    fn metrics(&self) -> CacheMetrics {
        self.inner.metrics()
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&mut self) -> usize {
        self.inner.clear()
    }
}

struct MaskArena<K, V>
where
    V: Cacheable<Key = K>,
{
    inner: GraphResourceCache<K, V>,
}

impl<K, V> MaskArena<K, V>
where
    K: Eq + std::hash::Hash + Clone + fmt::Debug,
    V: Cacheable<Key = K>,
{
    fn new(cache_name: &'static str) -> Self {
        Self {
            inner: GraphResourceCache::new(cache_name),
        }
    }

    fn get_or_create(
        &mut self,
        key: K,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut V, MetalError> {
        self.inner.get_or_create(key, explicit_device, default_device)
    }

    fn metrics(&self) -> CacheMetrics {
        self.inner.metrics()
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&mut self) -> usize {
        self.inner.clear()
    }
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::MTLDevice;

    use super::*;
    use crate::{cacheable::Cacheable, error::MetalError};

    // Mock cacheable resource for testing
    #[derive(Clone)]
    struct MockCacheableResource {
        value: i32,
    }

    impl Cacheable for MockCacheableResource {
        type Key = i32;

        fn cache_key(&self) -> Self::Key {
            self.value
        }

        fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
            Ok(MockCacheableResource { value: *key })
        }
    }

    #[test]
    fn test_cache_metrics_tracking() {
        let mut cache: FxHashMap<i32, CacheEntry<MockCacheableResource>> = FxHashMap::default();
        let mut counters = CacheCounters::default();
        let device: Option<&Retained<ProtocolObject<dyn MTLDevice>>> = None;
        let default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>> = None;

        // Test miss - create new resource
        let resource =
            ResourceCache::get_or_create_resource::<MockCacheableResource>(&mut cache, 42, device, default_device, &mut counters, "test")
                .unwrap();
        assert_eq!(resource.value, 42);
        assert_eq!(counters.misses, 1);
        assert_eq!(counters.hits, 0);

        // Test hit - get existing resource
        let resource =
            ResourceCache::get_or_create_resource::<MockCacheableResource>(&mut cache, 42, device, default_device, &mut counters, "test")
                .unwrap();
        assert_eq!(resource.value, 42);
        assert_eq!(counters.misses, 1);
        assert_eq!(counters.hits, 1);

        // Test another miss
        let resource =
            ResourceCache::get_or_create_resource::<MockCacheableResource>(&mut cache, 24, device, default_device, &mut counters, "test")
                .unwrap();
        assert_eq!(resource.value, 24);
        assert_eq!(counters.misses, 2);
        assert_eq!(counters.hits, 1);

        let metrics = metrics_for_entries(&cache, &counters);
        assert_eq!(metrics.max_entry_reuse_count, Some(1));
        assert!(metrics.oldest_entry_age.is_some());
    }

    #[test]
    fn test_resource_cache_get_stats() {
        let resource_cache = ResourceCache::new();

        // Initially, all caches should be empty
        let stats = resource_cache.get_stats();
        assert_eq!(stats.gemm.size, 0);
        assert_eq!(stats.descriptor.size, 0);
        assert_eq!(stats.softmax.size, 0);
        assert_eq!(stats.sdpa.size, 0);
        assert_eq!(stats.mpsgraph_sdpa.size, 0);

        // The hit/miss counts should also be zero initially
        assert_eq!(stats.gemm.hits, 0);
        assert_eq!(stats.gemm.misses, 0);
        assert_eq!(stats.mpsgraph_sdpa.hits, 0);
        assert_eq!(stats.mpsgraph_sdpa.misses, 0);
    }

    #[test]
    fn test_cache_metrics_snapshot_from_resource_cache() {
        use crate::resource_cache_metrics::CacheMetricsSnapshot;

        let resource_cache = ResourceCache::new();
        let snapshot: CacheMetricsSnapshot = (&resource_cache).into();

        // All metrics should be zero initially
        assert_eq!(snapshot.gemm.size, 0);
        assert_eq!(snapshot.descriptor.size, 0);
        assert_eq!(snapshot.softmax.size, 0);
        assert_eq!(snapshot.sdpa.size, 0);
        assert_eq!(snapshot.mpsgraph_sdpa.size, 0);
        assert_eq!(snapshot.mpsgraph_fused.size, 0);

        assert_eq!(snapshot.gemm.hits, 0);
        assert_eq!(snapshot.gemm.misses, 0);
        assert_eq!(snapshot.mpsgraph_sdpa.hits, 0);
        assert_eq!(snapshot.mpsgraph_sdpa.misses, 0);
        assert_eq!(snapshot.mpsgraph_fused.hits, 0);
        assert_eq!(snapshot.mpsgraph_fused.misses, 0);
    }
}
