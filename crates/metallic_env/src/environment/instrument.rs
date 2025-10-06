//! Instrumentation-specific environment variable identifiers.

use super::EnvVar;

/// Instrumentation-specific environment variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InstrumentEnvVar {
    /// Controls the global log level for the instrumentation pipeline.
    LogLevel,
    /// Path where metrics should be persisted as JSONL.
    MetricsJsonlPath,
    /// Enables console metrics emission when set to a truthy value.
    MetricsConsole,
}

impl InstrumentEnvVar {
    /// Obtain the canonical environment variable key for the identifier.
    pub const fn key(self) -> &'static str {
        match self {
            InstrumentEnvVar::LogLevel => "METALLIC_LOG_LEVEL",
            InstrumentEnvVar::MetricsJsonlPath => "METALLIC_METRICS_JSONL_PATH",
            InstrumentEnvVar::MetricsConsole => "METALLIC_METRICS_CONSOLE",
        }
    }

    /// Convert into the unscoped [`EnvVar`] variant.
    pub const fn into_env(self) -> EnvVar {
        EnvVar::Instrument(self)
    }
}
