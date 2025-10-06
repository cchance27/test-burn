//! Unified instrumentation module scaffolding the upcoming metrics system.

pub mod config;
pub mod environment;
pub mod event;
pub mod exporters;
pub mod gpu_profiler;
pub mod macros;
pub mod prelude;
pub mod recorder;

pub use config::{AppConfig, AppConfigError};
pub use event::MetricEvent;
pub use gpu_profiler::{GpuProfiler, GpuProfilerScope};
pub use recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer};

#[cfg(test)]
mod tests;
