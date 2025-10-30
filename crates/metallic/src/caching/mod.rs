//! Unified caching infrastructure for metallic.
//!
//! This module exposes building blocks for cacheable kernel resources and the
//! type-erased registry used by the metallic runtime.  The goal is to provide
//! a single entry-point for every cache-related primitive so that new kernels
//! can plug into the system without editing global cache structures.

pub mod entry;
pub mod metrics;
pub mod registry;
pub mod traits;

pub use entry::{CacheEntry, CacheLifetimeSummary, EntryMetadata};
pub use metrics::{CacheCounters, CacheEvent, CacheEventKind, CacheMetrics};
pub use registry::{AnyCache, CacheRegistry};
pub use traits::CacheableKernel;
