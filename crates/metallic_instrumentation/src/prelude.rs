//! Convenience re-exports for instrumentation consumers.

pub use chrono::{DateTime, Utc};
pub use metallic_env::{
    EnvVar, EnvVarError, EnvVarGuard, Environment, InstrumentEnvVar, environment::instrument::{ENABLE_PROFILING_VAR, LOG_LEVEL_VAR, METRICS_CONSOLE_VAR, METRICS_JSONL_PATH_VAR}
};
pub use serde_json;
pub use tracing::{Level, info, info_span, subscriber};
pub use tracing_subscriber::{self, layer::SubscriberExt};

pub use crate::{
    async_recorder::{AsyncMetricRecorder, MetricQueue}, config::{AppConfig, AppConfigError, ProfilingOverrideGuard, reset_app_config_for_tests}, event::MetricEvent, exporters::{ChannelExporter, ConsoleExporter, JsonlExporter}, macros::{MetricQueueBypassGuard, get_metric_queue, init_metric_queue, metric_queue_bypass_active, with_metric_queue_bypass}, record_metric, recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer}
};
