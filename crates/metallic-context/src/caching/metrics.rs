use std::{fmt, time::Duration};

use super::entry::{CacheEntry, CacheLifetimeSummary, summarise_lifetimes};

/// Summary of the most recent cache interaction.
#[derive(Clone, Debug)]
pub struct CacheEvent {
    pub kind: CacheEventKind,
    pub cache: &'static str,
    pub detail: String,
}

impl CacheEvent {
    #[inline]
    pub fn new(kind: CacheEventKind, cache: &'static str, detail: String) -> Self {
        Self { kind, cache, detail }
    }
}

impl fmt::Display for CacheEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} -> {}", self.cache, self.kind, self.detail)
    }
}

/// The type of cache event that most recently occurred for a cache.
#[derive(Clone, Copy, Debug)]
pub enum CacheEventKind {
    Hit,
    MissCreate,
    Cleared,
    Evicted,
}

impl fmt::Display for CacheEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hit => f.write_str("hit"),
            Self::MissCreate => f.write_str("miss-create"),
            Self::Cleared => f.write_str("cleared"),
            Self::Evicted => f.write_str("evicted"),
        }
    }
}

/// Internal counters used to build cache statistics.
#[derive(Clone, Debug, Default)]
pub struct CacheCounters {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evictions: u64,
    pub(crate) last_event: Option<CacheEvent>,
}

impl CacheCounters {
    #[inline]
    pub fn record_hit(&mut self, cache: &'static str, detail: String) {
        self.hits = self.hits.saturating_add(1);
        self.last_event = Some(CacheEvent::new(CacheEventKind::Hit, cache, detail));
    }

    #[inline]
    pub fn record_miss(&mut self, cache: &'static str, detail: String) {
        self.misses = self.misses.saturating_add(1);
        self.last_event = Some(CacheEvent::new(CacheEventKind::MissCreate, cache, detail));
    }

    #[inline]
    pub fn record_clear(&mut self, cache: &'static str, reason: &str, evicted: u64) {
        self.evictions = self.evictions.saturating_add(evicted);
        self.last_event = Some(CacheEvent::new(CacheEventKind::Cleared, cache, reason.to_string()));
    }

    #[inline]
    pub fn record_eviction(&mut self, cache: &'static str, detail: String, count: u64) {
        self.evictions = self.evictions.saturating_add(count);
        self.last_event = Some(CacheEvent::new(CacheEventKind::Evicted, cache, detail));
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

/// Statistics about an individual cache.
#[derive(Clone, Debug, Default)]
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
    #[inline]
    pub fn from_parts(size: usize, counters: &CacheCounters, lifetime: CacheLifetimeSummary) -> Self {
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

#[inline]
pub fn metrics_for_entries<K, V>(entries: &rustc_hash::FxHashMap<K, CacheEntry<V>>, counters: &CacheCounters) -> CacheMetrics
where
    K: std::hash::Hash + Eq,
{
    let lifetime = summarise_lifetimes(entries.values());
    CacheMetrics::from_parts(entries.len(), counters, lifetime)
}
