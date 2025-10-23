//! Developer-facing macros for emitting structured metric events with async support.
use std::sync::{
    OnceLock, atomic::{AtomicUsize, Ordering}
};

use crate::MetricQueue;

/// Global queue for async metric recording.
/// This is set during initialization and allows the macro to be zero-cost.
pub static METRIC_QUEUE: OnceLock<MetricQueue> = OnceLock::new();

static METRIC_QUEUE_BYPASS_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Guard that routes `record_metric_async!` through the synchronous fallback on the current thread.
pub struct MetricQueueBypassGuard;

impl Default for MetricQueueBypassGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricQueueBypassGuard {
    /// Engage the bypass for the current thread.
    pub fn new() -> Self {
        METRIC_QUEUE_BYPASS_COUNT.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for MetricQueueBypassGuard {
    fn drop(&mut self) {
        let result = METRIC_QUEUE_BYPASS_COUNT.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |count| count.checked_sub(1));
        debug_assert!(result.is_ok(), "MetricQueueBypassGuard underflow detected");
    }
}

/// Execute `f` with the metric queue bypass enabled on the current thread.
pub fn with_metric_queue_bypass<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let guard = MetricQueueBypassGuard::new();
    let result = f();
    drop(guard);
    result
}

#[doc(hidden)]
pub fn metric_queue_bypass_active() -> bool {
    METRIC_QUEUE_BYPASS_COUNT.load(Ordering::Acquire) > 0
}

#[macro_export]
macro_rules! record_metric {
    ($event:expr) => {{
        if let Ok(__metric_json) = serde_json::to_string(&$event) {
            tracing::event!(
                target: "metrics",
                tracing::Level::INFO,
                metric = %__metric_json
            );
        }
    }};
}

/// Emit a `MetricEvent` using the async recorder with lock-free queue.
/// This provides zero-overhead metric recording when metrics are enabled.
#[macro_export]
macro_rules! record_metric_async {
    ($event:expr) => {{
        use $crate::METRIC_QUEUE;
        let __event = $event;
        if !$crate::macros::metric_queue_bypass_active() {
            if let Some(queue) = METRIC_QUEUE.get() {
                queue.push(__event);
            } else {
                // Fall back to synchronous emission so tests and lightweight setups continue to observe metrics.
                $crate::record_metric!(__event);
            }
        } else {
            $crate::record_metric!(__event);
        }
    }};
}

/// Convenience macro for recording GGUF file memory-mapped usage.
#[macro_export]
macro_rules! record_gguf_mmap {
    ($size_bytes:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::GgufFileMmap { size_bytes: $size_bytes });
    };
}

/// Convenience macro for recording model weights memory usage.
#[macro_export]
macro_rules! record_model_weights {
    ($total_bytes:expr, $breakdown:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::ModelWeights {
            total_bytes: $total_bytes,
            breakdown: $breakdown,
        });
    };
}

/// Convenience macro for recording host memory usage.
#[macro_export]
macro_rules! record_host_memory {
    ($total_bytes:expr, $tensor_pool_reserved:expr, $kv_pool_reserved:expr, $kv_pool_used:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::HostMemory {
            total_bytes: $total_bytes,
            tensor_pool_reserved_bytes: $tensor_pool_reserved,
            kv_pool_reserved_bytes: $kv_pool_reserved,
            kv_pool_used_bytes: $kv_pool_used,
        });
    };
}

/// Convenience macro for recording forward step memory usage.
#[macro_export]
macro_rules! record_forward_step {
    ($total_bytes:expr, $breakdown:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::ForwardStep {
            total_bytes: $total_bytes,
            breakdown: $breakdown,
        });
    };
}

/// Convenience macro for recording forward block memory usage.
#[macro_export]
macro_rules! record_forward_block {
    ($block_index:expr, $block_name:expr, $breakdown:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::ForwardBlock {
            block_index: $block_index,
            block_name: $block_name,
            breakdown: $breakdown,
        });
    };
}

/// Convenience macro for recording tensor memory usage.
#[macro_export]
macro_rules! record_tensor_memory {
    ($total_bytes:expr, $tensor_count:expr, $breakdown:expr) => {
        $crate::record_metric_async!($crate::event::MetricEvent::TensorMemory {
            total_bytes: $total_bytes,
            tensor_count: $tensor_count,
            breakdown: $breakdown,
        });
    };
}

/// Initialize the global metric queue for async recording.
pub fn init_metric_queue(queue: MetricQueue) {
    if METRIC_QUEUE.set(queue).is_ok() {
        // Successfully set the queue
    } else {
        panic!("METRIC_QUEUE already initialized");
    }
}

/// Get a reference to the global metric queue.
pub fn get_metric_queue() -> Option<&'static MetricQueue> {
    METRIC_QUEUE.get()
}
