//! Canonical metric event definitions for the unified instrumentation system.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// Captures resource cache utilisation metrics.
    ResourceCacheAccess { cache_key: String, hit: bool, bytes: u64 },
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
}
