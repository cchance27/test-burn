//! Memory collection and reporting functionality for runtime memory statistics.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::{event::MetricEvent, record_metric_async};

/// Memory collector for gathering runtime memory statistics from Metal and CPU.
pub struct MemoryCollector {
    /// Whether memory collection is enabled.
    enabled: bool,
}

impl MemoryCollector {
    /// Create a new memory collector.
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Enable or disable memory collection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if memory collection is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Collect GGUF file memory-mapped size.
    pub fn collect_gguf_mmap(&self, size_bytes: u64) {
        if !self.enabled {
            return;
        }

        record_metric_async!(MetricEvent::GgufFileMmap { size_bytes });
    }

    /// Collect model weights memory usage with detailed breakdown.
    pub fn collect_model_weights(&self, total_bytes: u64, breakdown: FxHashMap<String, u64>) {
        if !self.enabled {
            return;
        }

        record_metric_async!(MetricEvent::ModelWeights { total_bytes, breakdown });
    }

    /// Collect host memory usage including tensor and KV pools.
    pub fn collect_host_memory(
        &self,
        total_bytes: u64,
        tensor_pool_reserved_bytes: u64,
        tensor_pool_used_bytes: u64,
        kv_pool_reserved_bytes: u64,
        kv_pool_used_bytes: u64,
        forward_pass_breakdown: BTreeMap<usize, (String, BTreeMap<String, u64>)>,
    ) {
        if !self.enabled {
            return;
        }

        record_metric_async!(MetricEvent::HostMemory {
            total_bytes,
            tensor_pool_reserved_bytes,
            tensor_pool_used_bytes,
            kv_pool_reserved_bytes,
            kv_pool_used_bytes,
            forward_pass_breakdown,
        });
    }

    /// Collect forward step memory usage with component breakdown.
    pub fn collect_forward_step(&self, total_bytes: u64, breakdown: FxHashMap<String, u64>) {
        if !self.enabled {
            return;
        }

        record_metric_async!(MetricEvent::ForwardStep { total_bytes, breakdown });
    }

    /// Collect general tensor memory usage statistics.
    pub fn collect_tensor_memory(&self, total_bytes: u64, tensor_count: u64, breakdown: FxHashMap<String, u64>) {
        if !self.enabled {
            return;
        }

        record_metric_async!(MetricEvent::TensorMemory {
            total_bytes,
            tensor_count,
            breakdown,
        });
    }
}

impl Default for MemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Global memory collector instance for convenience.
pub fn global_memory_collector() -> &'static MemoryCollector {
    static INSTANCE: std::sync::OnceLock<MemoryCollector> = std::sync::OnceLock::new();
    INSTANCE.get_or_init(MemoryCollector::new)
}
