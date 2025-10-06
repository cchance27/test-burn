//! Unified instrumentation module scaffolding the upcoming metrics system.

pub mod config;
pub mod event;
pub mod exporters;
pub mod macros;
pub mod prelude;
pub mod recorder;

pub use config::{AppConfig, AppConfigError};
pub use event::MetricEvent;
pub use recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer};

#[cfg(test)]
mod tests;
