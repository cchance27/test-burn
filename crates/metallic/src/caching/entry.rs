use std::time::{Duration, Instant};

/// Metadata captured for each cache entry so we can expose detailed metrics.
#[derive(Clone, Debug)]
pub struct EntryMetadata {
    pub(crate) created_at: Instant,
    pub(crate) last_used_at: Instant,
    pub(crate) reuse_count: u64,
}

impl EntryMetadata {
    #[inline]
    pub fn new(now: Instant) -> Self {
        Self {
            created_at: now,
            last_used_at: now,
            reuse_count: 0,
        }
    }

    #[inline]
    pub fn touch(&mut self, now: Instant) {
        self.last_used_at = now;
        self.reuse_count = self.reuse_count.saturating_add(1);
    }
}

/// Cached value plus associated metadata.
#[derive(Clone, Debug)]
pub struct CacheEntry<V> {
    pub(crate) value: V,
    pub(crate) metadata: EntryMetadata,
}

impl<V> CacheEntry<V> {
    #[inline]
    pub fn new(value: V, now: Instant) -> Self {
        Self {
            value,
            metadata: EntryMetadata::new(now),
        }
    }

    #[inline]
    pub fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

/// Aggregated lifetime statistics used while building cache metrics.
#[derive(Clone, Debug, Default)]
pub struct CacheLifetimeSummary {
    pub(crate) oldest_entry_age: Option<Duration>,
    pub(crate) newest_entry_age: Option<Duration>,
    pub(crate) longest_idle: Option<Duration>,
    pub(crate) shortest_idle: Option<Duration>,
    pub(crate) max_reuse_count: Option<u64>,
}

impl CacheLifetimeSummary {
    #[inline]
    pub fn observe(&mut self, metadata: &EntryMetadata, now: Instant) {
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

#[inline]
pub fn summarise_lifetimes<'a, V: 'a>(entries: impl Iterator<Item = &'a CacheEntry<V>>) -> CacheLifetimeSummary {
    let now = Instant::now();
    let mut summary = CacheLifetimeSummary::default();
    for entry in entries {
        summary.observe(&entry.metadata, now);
    }
    summary
}
