//! Canonical metric event definitions for the unified instrumentation system.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Structured, type-safe metric events emitted by the instrumentation layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MetricEvent {
    /// Indicates that a GPU kernel has been dispatched and is now in-flight.
    GpuKernelDispatched {
        kernel_name: String,
        /// A unique name for this specific operation instance.
        op_name: String,
        thread_groups: (u32, u32, u32),
    },
    /// Generated when the GPU reports an operation has completed.
    GpuOpCompleted {
        op_name: String,
        /// The backend that executed the operation (e.g., "Mlx", "Mps").
        backend: String,
        duration_us: u64,
    },
    /// Timing data for internal kernels invoked by frameworks such as MPS.
    InternalKernelCompleted {
        parent_op_name: String,
        internal_kernel_name: String,
        duration_us: u64,
    },
    /// Records which backend executed a kernel when multiple options are available.
    KernelBackendSelected { op_name: String, backend: String, reason: String },
    /// Captures resource cache utilisation metrics.
    ResourceCacheAccess { cache_key: String, hit: bool, bytes: u64 },
    /// Periodic summary of a resource cache (e.g., mpsgraph_sdpa) for dashboards.
    ResourceCacheSummary {
        cache: String,
        hits: u64,
        misses: u64,
        hit_rate: f64,
        size: u64,
    },
    /// Cache eviction event with detailed metadata.
    CacheEviction {
        /// Name of the cache that performed eviction.
        cache: String,
        /// Eviction strategy that triggered this event.
        strategy: String,
        /// Number of entries evicted.
        count: u64,
        /// Reason for eviction (e.g., "size_limit_exceeded", "idle_timeout", "manual_clear").
        reason: String,
        /// Cache size after eviction.
        size_after: u64,
    },
    /// Memory-mapped file usage for GGUF models.
    GgufFileMmap {
        /// Size of the memory-mapped file in bytes.
        size_bytes: u64,
    },
    /// Model weights memory usage with detailed breakdown.
    ModelWeights {
        /// Total size of model weights in bytes.
        total_bytes: u64,
        /// Breakdown of weights by category.
        breakdown: FxHashMap<String, u64>,
    },
    /// Host memory usage including tensor and KV pools.
    HostMemory {
        /// Total host memory in bytes.
        total_bytes: u64,
        /// Reserved tensor pool memory in bytes.
        tensor_pool_reserved_bytes: u64,
        /// Used tensor pool memory in bytes.
        tensor_pool_used_bytes: u64,
        /// Reserved KV pool memory in bytes.
        kv_pool_reserved_bytes: u64,
        /// Used KV pool memory in bytes.
        kv_pool_used_bytes: u64,
        /// Memory usage breakdown of the forward pass.
        forward_pass_breakdown: BTreeMap<usize, (String, BTreeMap<String, u64>)>,
    },
    /// Forward pass step memory usage with per-block breakdown.
    ForwardStep {
        /// Total forward step memory in bytes.
        total_bytes: u64,
        /// Memory usage breakdown by component.
        breakdown: FxHashMap<String, u64>,
    },
    /// General tensor memory usage reporting.
    TensorMemory {
        /// Total tensor memory in bytes.
        total_bytes: u64,
        /// Number of active tensors.
        tensor_count: u64,
        /// Memory usage by tensor category.
        breakdown: FxHashMap<String, u64>,
    },
    /// Tensor preparation cache statistics.
    TensorPreparationStats {
        /// Number of times tensor preparation was skipped due to cache hit
        cache_hits: u64,
        /// Number of times tensor preparation was performed (cache miss)
        cache_misses: u64,
        /// Total time spent in preparation operations (microseconds)
        total_preparation_time_us: u64,
        /// Estimated time saved by cache hits (microseconds)
        estimated_time_saved_us: u64,
        /// Hit rate as a percentage (0-100)
        hit_rate: f64,
    },
}
