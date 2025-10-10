//! Developer-facing macros for emitting structured metric events with async support.
use crate::MetricQueue;
use std::sync::OnceLock;

/// Global queue for async metric recording.
/// This is set during initialization and allows the macro to be zero-cost.
pub static METRIC_QUEUE: OnceLock<MetricQueue> = OnceLock::new();

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
        // This is a compile-time check - if METRIC_QUEUE is not set, this becomes a no-op
        if let Some(queue) = METRIC_QUEUE.get() {
            // Lock-free push - this is extremely fast and doesn't block
            queue.push($event);
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
