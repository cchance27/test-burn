//! Convenience re-exports for instrumentation consumers.

pub use crate::metallic::instrument::config::{AppConfig, AppConfigError};
pub use crate::metallic::instrument::event::MetricEvent;
pub use crate::metallic::instrument::exporters::{ChannelExporter, ConsoleExporter, JsonlExporter};
pub use crate::metallic::instrument::recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer};
pub use crate::record_metric;
