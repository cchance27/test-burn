//! Developer-facing macros for emitting structured metric events.

use tracing::Level;

/// Emit a `MetricEvent` through the tracing infrastructure.
#[macro_export]
macro_rules! record_metric {
    ($event:expr) => {{
        if tracing::enabled!(target: "metrics", Level::INFO) {
            if let Ok(__metric_json) = serde_json::to_string(&$event) {
                tracing::event!(
                    target: "metrics",
                    Level::INFO,
                    metric = %__metric_json
                );
            }
        }
    }};
}
