use std::{
    any::{Any, TypeId}, collections::hash_map::Entry, marker::PhantomData
};

use metallic_instrumentation::{MetricEvent, prelude::info_span, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use rustc_hash::FxHashMap;

use super::{
    entry::CacheEntry, eviction::EvictionPolicy, metrics::{CacheCounters, CacheMetrics, metrics_for_entries}, traits::CacheableKernel
};
use crate::error::MetalError;

/// Trait object used to store caches for different kernel implementations inside
/// the registry without knowing their concrete types.
pub trait AnyCache: Any {
    fn cache_name(&self) -> &'static str;
    fn clear(&mut self) -> usize;
    fn metrics(&self) -> CacheMetrics;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

trait TypedSlot: Any {
    fn clear_slot(&mut self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Slot types stored alongside kernel caches within the registry.
pub trait CacheRegistrySlot: Any {
    fn clear_slot(&mut self);
}

struct SlotWrapper<T: CacheRegistrySlot> {
    inner: T,
}

impl<T: CacheRegistrySlot> SlotWrapper<T> {
    fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: CacheRegistrySlot> TypedSlot for SlotWrapper<T> {
    fn clear_slot(&mut self) {
        self.inner.clear_slot();
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

struct KernelCache<K: CacheableKernel> {
    entries: FxHashMap<K::Key, CacheEntry<K::CachedResource>>,
    counters: CacheCounters,
    eviction_policy: EvictionPolicy,
    _marker: PhantomData<K>,
}

impl<K: CacheableKernel> KernelCache<K> {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            counters: CacheCounters::default(),
            eviction_policy: EvictionPolicy::default(),
            _marker: PhantomData,
        }
    }

    #[inline]
    fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    fn cache_name() -> &'static str {
        K::CACHE_NAME
    }

    fn get_or_create(
        &mut self,
        params: &K::Params,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        default_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut K::CachedResource, MetalError> {
        let device = explicit_device.or(default_device);
        let cache_name = Self::cache_name();
        let span = info_span!("cache_get_or_create", cache = cache_name);
        let _enter = span.enter();

        let key = K::create_cache_key(params);
        let now = std::time::Instant::now();

        // Check if key exists first (cache hit case)
        let is_cache_hit = self.entries.contains_key(&key);

        if is_cache_hit {
            let cache_size = self.entries.len();
            let detail = format!("{:?}", key);
            self.counters.record_hit(cache_name, detail.clone());
            record_metric_async!(MetricEvent::ResourceCacheAccess {
                cache_key: format!("{}:{}", cache_name, detail),
                hit: true,
                bytes: 0,
            });
            let ops = self.counters.hits().saturating_add(self.counters.misses());
            if ops.is_multiple_of(100) {
                let hit_rate = if ops == 0 {
                    0.0
                } else {
                    (self.counters.hits() as f64 / ops as f64) * 100.0
                };
                record_metric_async!(MetricEvent::ResourceCacheSummary {
                    cache: cache_name.to_string(),
                    hits: self.counters.hits(),
                    misses: self.counters.misses(),
                    hit_rate,
                    size: cache_size as u64,
                });
            }
            // Get mutable reference after all immutable operations are done
            let entry = self.entries.get_mut(&key).expect("key must exist");
            entry.metadata.touch(now);
            return Ok(entry.value_mut());
        }

        // Cache miss - perform eviction before insertion to maintain size limits
        self.maybe_evict_entries();

        let cache_size_before_insert = self.entries.len();

        match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let detail = format!("{:?}", entry.key());
                self.counters.record_hit(cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", cache_name, detail),
                    hit: true,
                    bytes: 0,
                });
                let ops = self.counters.hits().saturating_add(self.counters.misses());
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (self.counters.hits() as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: cache_name.to_string(),
                        hits: self.counters.hits(),
                        misses: self.counters.misses(),
                        hit_rate,
                        size: cache_size_before_insert as u64,
                    });
                }
                let entry_ref = entry.into_mut();
                entry_ref.metadata.touch(now);
                Ok(entry_ref.value_mut())
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let detail = format!("{:?}", entry.key());
                self.counters.record_miss(cache_name, detail.clone());
                record_metric_async!(MetricEvent::ResourceCacheAccess {
                    cache_key: format!("{}:{}", cache_name, detail),
                    hit: false,
                    bytes: 0,
                });
                let resource = K::create_cached_resource(entry.key(), device)?;
                let entry_ref = entry.insert(CacheEntry::new(resource, now));
                let ops = self.counters.hits().saturating_add(self.counters.misses());
                if ops.is_multiple_of(100) {
                    let hit_rate = if ops == 0 {
                        0.0
                    } else {
                        (self.counters.hits() as f64 / ops as f64) * 100.0
                    };
                    record_metric_async!(MetricEvent::ResourceCacheSummary {
                        cache: cache_name.to_string(),
                        hits: self.counters.hits(),
                        misses: self.counters.misses(),
                        hit_rate,
                        size: cache_size_before_insert as u64,
                    });
                }
                Ok(entry_ref.value_mut())
            }
        }
    }

    fn clear(&mut self) -> usize {
        let evicted = self.entries.len();
        self.entries.clear();
        evicted
    }

    fn metrics(&self) -> CacheMetrics {
        metrics_for_entries(&self.entries, &self.counters)
    }

    /// Check if eviction should be triggered based on the configured policy.
    /// Returns true if we're at or over the size limit.
    #[inline]
    fn should_evict_for_size(&self) -> bool {
        use super::eviction::EvictionStrategy;

        match self.eviction_policy.strategy {
            EvictionStrategy::None => false,
            EvictionStrategy::SizeLimitedLru { max_entries } => self.entries.len() >= max_entries,
            _ => {
                if let Some(max) = self.eviction_policy.max_entries {
                    self.entries.len() >= max
                } else {
                    false
                }
            }
        }
    }

    /// Perform eviction based on the configured policy.
    /// This is called before cache miss insertions to maintain size limits.
    fn maybe_evict_entries(&mut self) {
        use super::eviction::EvictionStrategy;

        if !self.eviction_policy.allows_eviction() {
            return;
        }

        let now = std::time::Instant::now();
        let cache_name = Self::cache_name();
        let min_entries = self.eviction_policy.min_entries;

        // First, handle idle timeout evictions (these run regardless of size)
        if let Some(max_idle) = self.eviction_policy.max_idle_duration {
            self.evict_idle(now, max_idle, cache_name);
        }

        // Then, handle size-based evictions - we need to evict BEFORE inserting
        // So check if we're at the limit (not over, since we haven't inserted yet)
        if self.should_evict_for_size() {
            match self.eviction_policy.strategy {
                EvictionStrategy::None => {}
                EvictionStrategy::Lru => {
                    // Evict one entry to make room
                    self.evict_lru(now, min_entries, cache_name);
                }
                EvictionStrategy::Fifo => {
                    // Evict one entry to make room
                    self.evict_fifo(min_entries, cache_name);
                }
                EvictionStrategy::IdleTimeout => {
                    // Already handled above
                }
                EvictionStrategy::SizeLimitedLru { max_entries } => {
                    // Evict entries to be under the limit (leaving room for the new one)
                    while self.entries.len() >= max_entries && self.entries.len() > min_entries {
                        self.evict_lru(now, min_entries, cache_name);
                    }
                }
            }
        }
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self, _now: std::time::Instant, min_entries: usize, cache_name: &'static str) {
        if self.entries.len() <= min_entries {
            return;
        }

        // Find the least recently used entry
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.metadata.last_used_at)
            .map(|(key, _)| key.clone());

        if let Some(key) = oldest_key {
            self.entries.remove(&key);
            self.counters.record_eviction(cache_name, format!("lru:{:?}", key), 1);

            record_metric_async!(MetricEvent::CacheEviction {
                cache: cache_name.to_string(),
                strategy: "lru".to_string(),
                count: 1,
                reason: "least_recently_used".to_string(),
                size_after: self.entries.len() as u64,
            });
        }
    }

    /// Evict the oldest entry by creation time (FIFO).
    fn evict_fifo(&mut self, min_entries: usize, cache_name: &'static str) {
        if self.entries.len() <= min_entries {
            return;
        }

        // Find the oldest entry by creation time
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.metadata.created_at)
            .map(|(key, _)| key.clone());

        if let Some(key) = oldest_key {
            self.entries.remove(&key);
            self.counters.record_eviction(cache_name, format!("fifo:{:?}", key), 1);

            record_metric_async!(MetricEvent::CacheEviction {
                cache: cache_name.to_string(),
                strategy: "fifo".to_string(),
                count: 1,
                reason: "first_in_first_out".to_string(),
                size_after: self.entries.len() as u64,
            });
        }
    }

    /// Evict all entries that have been idle for longer than max_idle.
    fn evict_idle(&mut self, now: std::time::Instant, max_idle: std::time::Duration, cache_name: &'static str) {
        let keys_to_evict: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| now.saturating_duration_since(entry.metadata.last_used_at) > max_idle)
            .map(|(key, _)| key.clone())
            .collect();

        let count = keys_to_evict.len();
        if count == 0 {
            return;
        }

        for key in keys_to_evict {
            self.entries.remove(&key);
        }

        self.counters
            .record_eviction(cache_name, format!("idle_timeout:{}ms", max_idle.as_millis()), count as u64);

        record_metric_async!(MetricEvent::CacheEviction {
            cache: cache_name.to_string(),
            strategy: "idle_timeout".to_string(),
            count: count as u64,
            reason: format!("idle_duration_exceeded_{}ms", max_idle.as_millis()),
            size_after: self.entries.len() as u64,
        });
    }
}

struct TypedCache<K: CacheableKernel> {
    inner: KernelCache<K>,
}

impl<K: CacheableKernel> TypedCache<K> {
    fn new() -> Self {
        Self { inner: KernelCache::new() }
    }
}

impl<K: CacheableKernel> AnyCache for TypedCache<K> {
    fn cache_name(&self) -> &'static str {
        KernelCache::<K>::cache_name()
    }

    fn clear(&mut self) -> usize {
        let evicted = self.inner.clear();
        self.inner
            .counters
            .record_clear(self.cache_name(), "registry clear", evicted as u64);
        evicted
    }

    fn metrics(&self) -> CacheMetrics {
        self.inner.metrics()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Top level cache registry keyed by the kernel type.
pub struct CacheRegistry {
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    caches: FxHashMap<TypeId, Box<dyn AnyCache>>,
    typed_slots: FxHashMap<TypeId, Box<dyn TypedSlot>>,
}

impl CacheRegistry {
    #[inline]
    pub fn with_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self::with_default_device(Some(device))
    }

    pub(crate) fn with_default_device(device: Option<Retained<ProtocolObject<dyn MTLDevice>>>) -> Self {
        Self {
            default_device: device,
            caches: FxHashMap::default(),
            typed_slots: FxHashMap::default(),
        }
    }

    fn typed_cache_mut<K: CacheableKernel>(&mut self) -> &mut KernelCache<K> {
        let type_id = TypeId::of::<K>();
        let entry = self
            .caches
            .entry(type_id)
            .or_insert_with(|| Box::new(TypedCache::<K>::new()) as Box<dyn AnyCache>);
        entry.as_any_mut().downcast_mut::<KernelCache<K>>().expect("cache type mismatch")
    }

    fn slot_entry_mut<T>(&mut self, type_id: TypeId, init: impl FnOnce() -> T) -> &mut dyn TypedSlot
    where
        T: CacheRegistrySlot + 'static,
    {
        match self.typed_slots.entry(type_id) {
            Entry::Occupied(entry) => entry.into_mut().as_mut(),
            Entry::Vacant(vacant) => {
                let wrapper: Box<dyn TypedSlot> = Box::new(SlotWrapper::new(init()));
                vacant.insert(wrapper).as_mut()
            }
        }
    }

    #[inline]
    pub fn get_or_create<K: CacheableKernel>(
        &mut self,
        params: &K::Params,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut K::CachedResource, MetalError> {
        let default_ptr = self
            .default_device
            .as_ref()
            .map(|device| device as *const Retained<ProtocolObject<dyn MTLDevice>>);
        let cache = self.typed_cache_mut::<K>();
        // SAFETY: `default_ptr` is derived from an immutable borrow taken before the
        // mutable borrow for `typed_cache_mut`. The underlying `Retained` lives for
        // the entire lifetime of `self`, so dereferencing the raw pointer is safe.
        let default = default_ptr.map(|ptr| unsafe { &*ptr });
        cache.get_or_create(params, explicit_device, default)
    }

    #[inline]
    pub fn clear(&mut self) {
        for cache in self.caches.values_mut() {
            cache.clear();
        }
    }

    #[inline]
    pub fn metrics<K: CacheableKernel>(&self) -> Option<CacheMetrics> {
        let type_id = TypeId::of::<K>();
        self.caches.get(&type_id).map(|cache| cache.metrics())
    }

    #[inline]
    pub fn metrics_by_name(&self) -> Vec<(&'static str, CacheMetrics)> {
        self.caches.values().map(|cache| (cache.cache_name(), cache.metrics())).collect()
    }

    /// Set the eviction policy for a specific kernel type.
    #[inline]
    pub fn set_eviction_policy<K: CacheableKernel>(&mut self, policy: EvictionPolicy) {
        let cache = self.typed_cache_mut::<K>();
        cache.set_eviction_policy(policy);
    }

    /// Get the current eviction policy for a specific kernel type.
    #[inline]
    pub fn eviction_policy<K: CacheableKernel>(&mut self) -> Option<&EvictionPolicy> {
        let type_id = TypeId::of::<K>();
        self.caches.get_mut(&type_id).map(|cache| {
            let cache_ptr = cache.as_any_mut() as *const _ as *const KernelCache<K>;
            unsafe { &(*cache_ptr).eviction_policy }
        })
    }

    #[inline]
    pub fn slot_mut<T, F>(&mut self, init: F) -> &mut T
    where
        T: CacheRegistrySlot + 'static,
        F: FnOnce() -> T,
    {
        let type_id = TypeId::of::<T>();
        let slot = self.slot_entry_mut(type_id, init);
        slot.as_any_mut().downcast_mut::<T>().expect("slot type mismatch")
    }

    #[inline]
    pub fn slot<T: CacheRegistrySlot + 'static>(&self) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.typed_slots.get(&type_id).and_then(|slot| slot.as_any().downcast_ref::<T>())
    }

    #[inline]
    pub fn clear_slot<T: CacheRegistrySlot + 'static>(&mut self) {
        if let Some(slot) = self.typed_slots.get_mut(&TypeId::of::<T>()) {
            slot.clear_slot();
        }
    }

    #[inline]
    pub fn slot_mut_existing<T: CacheRegistrySlot + 'static>(&mut self) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.typed_slots
            .get_mut(&type_id)
            .and_then(|slot| slot.as_any_mut().downcast_mut::<T>())
    }
}

impl Default for CacheRegistry {
    #[inline]
    fn default() -> Self {
        Self::with_default_device(None)
    }
}

#[cfg(test)]
#[path = "eviction_tests.rs"]
mod eviction_tests;
