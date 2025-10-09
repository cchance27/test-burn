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

/// Initialize the global metric queue for async recording.
pub fn init_metric_queue(queue: MetricQueue) {
    if METRIC_QUEUE.set(queue).is_err() {
        panic!("METRIC_QUEUE already initialized");
    }
}

/// Get a reference to the global metric queue.
pub fn get_metric_queue() -> Option<&'static MetricQueue> {
    METRIC_QUEUE.get()
}
