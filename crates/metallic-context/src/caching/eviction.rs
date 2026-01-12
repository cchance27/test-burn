//! Eviction policies and strategies for cache management.
//!
//! This module provides configurable cache eviction to prevent unbounded memory growth.
//! Eviction events are automatically logged via the metallic_instrumentation system.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Strategy for selecting which entries to evict when cache pressure is detected.
///
/// Each strategy is designed for different use cases:
/// - `None`: Suitable for caches with naturally bounded working sets
/// - `Lru`: Best for uniform access patterns where recent = relevant
/// - `Fifo`: Simple time-based eviction, good for predictable lifecycles
/// - `IdleTimeout`: Automatically removes stale entries
/// - `SizeLimitedLru`: Enforces hard limits on cache size with LRU fallback
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
#[derive(Default)]
pub enum EvictionStrategy {
    /// No eviction - cache grows unbounded (default behavior).
    /// Use when the working set is naturally bounded or memory is not a concern.
    #[default]
    None,

    /// Evict least recently used entries.
    /// Optimizes for temporal locality - entries accessed recently are kept.
    Lru,

    /// Evict oldest entries by creation time (First-In, First-Out).
    /// Simpler than LRU, useful when all entries have similar access patterns.
    Fifo,

    /// Evict entries that haven't been accessed within the idle duration.
    /// Useful for automatically cleaning up entries that are no longer needed.
    IdleTimeout,

    /// Size-limited LRU with a specific maximum entry count.
    /// Provides strict memory bounds with intelligent eviction.
    SizeLimitedLru { max_entries: usize },
}


/// Configuration for cache eviction behavior.
///
/// Policies can be combined - for example, size limits with idle timeouts.
/// The policy is evaluated on every cache insertion.
///
/// # Examples
///
/// ```
/// use metallic::caching::EvictionPolicy;
/// use std::time::Duration;
///
/// // Simple size limit
/// let policy = EvictionPolicy::size_limited_lru(1000);
///
/// // Idle timeout
/// let policy = EvictionPolicy::idle_timeout(Duration::from_secs(300));
///
/// // Combined: size limit + idle timeout
/// let policy = EvictionPolicy::hybrid(1000, Duration::from_secs(300));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvictionPolicy {
    /// Primary eviction strategy.
    pub strategy: EvictionStrategy,

    /// Maximum number of entries before eviction triggers (None = unlimited).
    /// Applies to all strategies except `SizeLimitedLru` which has its own limit.
    pub max_entries: Option<usize>,

    /// Maximum idle duration before entry is eligible for eviction.
    /// Applied in addition to the primary strategy.
    pub max_idle_duration: Option<Duration>,

    /// Minimum entries to keep (prevents over-eviction).
    /// Useful for maintaining a "warm" cache even under pressure.
    pub min_entries: usize,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            strategy: EvictionStrategy::None,
            max_entries: None,
            max_idle_duration: None,
            min_entries: 0,
        }
    }
}

impl EvictionPolicy {
    /// Create a policy with no eviction (unbounded cache).
    #[inline]
    pub fn none() -> Self {
        Self::default()
    }

    /// Create a size-limited LRU policy with a maximum number of entries.
    ///
    /// When the cache reaches `max_entries`, the least recently used entry is evicted.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries before eviction triggers
    #[inline]
    pub fn size_limited_lru(max_entries: usize) -> Self {
        Self {
            strategy: EvictionStrategy::SizeLimitedLru { max_entries },
            max_entries: Some(max_entries),
            max_idle_duration: None,
            min_entries: 0,
        }
    }

    /// Create an idle timeout policy.
    ///
    /// Entries that haven't been accessed within `max_idle` are eligible for eviction.
    /// Eviction is checked on every cache operation.
    ///
    /// # Arguments
    /// * `max_idle` - Maximum duration an entry can remain idle before eviction
    #[inline]
    pub fn idle_timeout(max_idle: Duration) -> Self {
        Self {
            strategy: EvictionStrategy::IdleTimeout,
            max_entries: None,
            max_idle_duration: Some(max_idle),
            min_entries: 0,
        }
    }

    /// Create a hybrid policy with both size limit and idle timeout.
    ///
    /// Entries are evicted if either condition is met:
    /// - Cache size exceeds `max_entries` (LRU eviction)
    /// - Entry has been idle for more than `max_idle`
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries
    /// * `max_idle` - Maximum idle duration
    #[inline]
    pub fn hybrid(max_entries: usize, max_idle: Duration) -> Self {
        Self {
            strategy: EvictionStrategy::SizeLimitedLru { max_entries },
            max_entries: Some(max_entries),
            max_idle_duration: Some(max_idle),
            min_entries: 0,
        }
    }

    /// Create a simple LRU policy without a hard size limit.
    ///
    /// Eviction only occurs when combined with `with_max_entries()`.
    #[inline]
    pub fn lru() -> Self {
        Self {
            strategy: EvictionStrategy::Lru,
            max_entries: None,
            max_idle_duration: None,
            min_entries: 0,
        }
    }

    /// Create a simple FIFO policy without a hard size limit.
    ///
    /// Eviction only occurs when combined with `with_max_entries()`.
    #[inline]
    pub fn fifo() -> Self {
        Self {
            strategy: EvictionStrategy::Fifo,
            max_entries: None,
            max_idle_duration: None,
            min_entries: 0,
        }
    }

    /// Set the maximum number of entries for this policy.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries before eviction triggers
    #[inline]
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = Some(max_entries);
        self
    }

    /// Set the maximum idle duration for this policy.
    ///
    /// # Arguments
    /// * `max_idle` - Maximum duration an entry can remain idle
    #[inline]
    pub fn with_max_idle_duration(mut self, max_idle: Duration) -> Self {
        self.max_idle_duration = Some(max_idle);
        self
    }

    /// Set the minimum number of entries to keep (prevents over-eviction).
    ///
    /// # Arguments
    /// * `min_entries` - Minimum entries to retain even under pressure
    #[inline]
    pub fn with_min_entries(mut self, min_entries: usize) -> Self {
        self.min_entries = min_entries;
        self
    }

    /// Check if this policy allows eviction.
    #[inline]
    pub fn allows_eviction(&self) -> bool {
        !matches!(self.strategy, EvictionStrategy::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_disables_eviction() {
        let policy = EvictionPolicy::default();
        assert_eq!(policy.strategy, EvictionStrategy::None);
        assert!(!policy.allows_eviction());
    }

    #[test]
    fn size_limited_lru_builder() {
        let policy = EvictionPolicy::size_limited_lru(100);
        assert!(matches!(policy.strategy, EvictionStrategy::SizeLimitedLru { max_entries: 100 }));
        assert_eq!(policy.max_entries, Some(100));
        assert!(policy.allows_eviction());
    }

    #[test]
    fn idle_timeout_builder() {
        let duration = Duration::from_secs(300);
        let policy = EvictionPolicy::idle_timeout(duration);
        assert_eq!(policy.strategy, EvictionStrategy::IdleTimeout);
        assert_eq!(policy.max_idle_duration, Some(duration));
        assert!(policy.allows_eviction());
    }

    #[test]
    fn hybrid_builder() {
        let duration = Duration::from_secs(600);
        let policy = EvictionPolicy::hybrid(200, duration);
        assert!(matches!(policy.strategy, EvictionStrategy::SizeLimitedLru { max_entries: 200 }));
        assert_eq!(policy.max_entries, Some(200));
        assert_eq!(policy.max_idle_duration, Some(duration));
        assert!(policy.allows_eviction());
    }

    #[test]
    fn fluent_builder_pattern() {
        let policy = EvictionPolicy::lru()
            .with_max_entries(500)
            .with_max_idle_duration(Duration::from_secs(120))
            .with_min_entries(10);

        assert_eq!(policy.strategy, EvictionStrategy::Lru);
        assert_eq!(policy.max_entries, Some(500));
        assert_eq!(policy.max_idle_duration, Some(Duration::from_secs(120)));
        assert_eq!(policy.min_entries, 10);
    }

    #[test]
    fn serialization_roundtrip() {
        let policy = EvictionPolicy::hybrid(1000, Duration::from_secs(300));
        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: EvictionPolicy = serde_json::from_str(&json).unwrap();

        assert_eq!(policy.max_entries, deserialized.max_entries);
        assert_eq!(policy.max_idle_duration, deserialized.max_idle_duration);
        assert_eq!(policy.min_entries, deserialized.min_entries);
    }
}
