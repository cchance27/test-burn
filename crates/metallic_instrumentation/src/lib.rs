//! Unified instrumentation module scaffolding the upcoming metrics system.

pub mod async_recorder;
pub mod config;
pub mod event;
pub mod exporters;
pub mod gpu_profiler;
pub mod macros;
pub mod prelude;
pub mod recorder;

pub use async_recorder::{AsyncMetricRecorder, MetricQueue};
pub use event::MetricEvent;
pub use gpu_profiler::GpuProfiler;
pub use macros::METRIC_QUEUE;

mod tests;
