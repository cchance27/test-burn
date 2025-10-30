use std::{
    any::{Any, TypeId}, marker::PhantomData
};

use metallic_instrumentation::{MetricEvent, prelude::info_span, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use rustc_hash::FxHashMap;

use super::{
    entry::CacheEntry, metrics::{CacheCounters, CacheMetrics, metrics_for_entries}, traits::CacheableKernel
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

struct KernelCache<K: CacheableKernel> {
    entries: FxHashMap<K::Key, CacheEntry<K::CachedResource>>,
    counters: CacheCounters,
    _marker: PhantomData<K>,
}

impl<K: CacheableKernel> KernelCache<K> {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            counters: CacheCounters::default(),
            _marker: PhantomData,
        }
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
        let cache_size_before_insert = self.entries.len();
        let now = std::time::Instant::now();

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
}

impl CacheRegistry {
    #[inline]
    pub fn new() -> Self {
        Self::with_default_device(None)
    }

    #[inline]
    pub fn with_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self::with_default_device(Some(device))
    }

    pub(crate) fn with_default_device(device: Option<Retained<ProtocolObject<dyn MTLDevice>>>) -> Self {
        Self {
            default_device: device,
            caches: FxHashMap::default(),
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
        self.caches.iter().map(|(_, cache)| (cache.cache_name(), cache.metrics())).collect()
    }
}
