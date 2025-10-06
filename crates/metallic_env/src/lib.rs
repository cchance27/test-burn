//! Shared process environment helpers for Metallic instrumentation.

pub mod environment;

pub use environment::guard::EnvVarGuard;
pub use environment::instrument::{
    InstrumentEnvVar, InstrumentLogLevel, InstrumentMetricsConsole, InstrumentMetricsJsonlPath, LOG_LEVEL, LogLevel, METRICS_CONSOLE,
    METRICS_JSONL_PATH, MetricsConsole, MetricsJsonlPath,
};
pub use environment::value::{EnvVarError, EnvVarFormatError, EnvVarParseError, TypedEnvVar, TypedEnvVarGuard};
pub use environment::{EnvVar, Environment};

#[cfg(test)]
pub use environment::{guard, instrument, value};
