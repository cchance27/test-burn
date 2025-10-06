//! Developer-facing macros for emitting structured metric events.

/// Emit a `MetricEvent` through the tracing infrastructure.
#[macro_export]
macro_rules! record_metric {
    ($event:expr) => {{
        if tracing::enabled!(target: "metrics", tracing::Level::INFO) {
            if let Ok(__metric_json) = serde_json::to_string(&$event) {
                tracing::event!(
                    target: "metrics",
                    tracing::Level::INFO,
                    metric = %__metric_json
                );
            }
        }
    }};
}
