//! Convenience re-exports for instrumentation consumers.

pub use crate::async_recorder::{AsyncMetricRecorder, MetricQueue};
pub use crate::config::{AppConfig, AppConfigError, ProfilingOverrideGuard, reset_app_config_for_tests};
pub use crate::event::MetricEvent;
pub use crate::exporters::{ChannelExporter, ConsoleExporter, JsonlExporter};
pub use crate::recorder::{EnrichedMetricEvent, MetricExporter, MetricsLayer};
pub use crate::{
    macros::{MetricQueueBypassGuard, get_metric_queue, init_metric_queue, metric_queue_bypass_active, with_metric_queue_bypass},
    record_metric,
};

pub use metallic_env::environment::instrument::{ENABLE_PROFILING_VAR, LOG_LEVEL_VAR, METRICS_CONSOLE_VAR, METRICS_JSONL_PATH_VAR};
pub use metallic_env::{EnvVar, EnvVarError, EnvVarGuard, Environment, InstrumentEnvVar};

pub use chrono::{DateTime, Utc};
pub use serde_json;
pub use tracing::{Level, info, info_span, subscriber};
pub use tracing_subscriber::{self, layer::SubscriberExt};
