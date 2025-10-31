//! Unified caching infrastructure for metallic.
//!
//! This module exposes building blocks for cacheable kernel resources and the
//! type-erased registry used by the metallic runtime.  The goal is to provide
//! a single entry-point for every cache-related primitive so that new kernels
//! can plug into the system without editing global cache structures.

pub mod entry;
pub mod eviction;
pub mod kv;
pub mod metrics;
pub mod registry;
pub mod resource;
pub mod resource_metrics;
pub mod tensor_preparation;
pub mod traits;

pub use entry::{CacheEntry, CacheLifetimeSummary, EntryMetadata};
pub use eviction::{EvictionPolicy, EvictionStrategy};
pub use kv::{KV_CACHE_POOL_MAX_BYTES, KvCacheEntry, KvCacheState, KvWritePlan};
pub use metrics::{CacheCounters, CacheEvent, CacheEventKind, CacheMetrics};
pub use registry::{AnyCache, CacheRegistry, CacheRegistrySlot};
pub use resource::{CacheStats, ResourceCache};
pub use resource_metrics::CacheMetricsSnapshot;
pub use tensor_preparation::{TensorPreparationCache, TensorPreparationMetrics};
pub use traits::CacheableKernel;
