//! Canonical metric event definitions for the unified instrumentation system.

use serde::{Deserialize, Serialize};

/// Structured, type-safe metric events emitted by the instrumentation layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MetricEvent {
    /// Indicates that a GPU kernel has been dispatched and is now in-flight.
    GpuKernelDispatched {
        kernel_name: &'static str,
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
}
