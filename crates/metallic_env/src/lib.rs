//! Shared process environment helpers for Metallic instrumentation.

pub mod environment;

pub use environment::guard::EnvVarGuard;
pub use environment::instrument::{
    ENABLE_PROFILING, ENABLE_PROFILING_VAR, FORCE_MATMUL_BACKEND, FORCE_MATMUL_BACKEND_VAR, InstrumentEnableProfiling, InstrumentEnvVar,
    InstrumentForceMatmulBackend, InstrumentLogLevel, InstrumentMetricsConsole, InstrumentMetricsJsonlPath, InstrumentSoftmaxBackend,
    LOG_LEVEL, LOG_LEVEL_VAR, METRICS_CONSOLE, METRICS_CONSOLE_VAR, METRICS_JSONL_PATH, METRICS_JSONL_PATH_VAR, SOFTMAX_BACKEND,
    SOFTMAX_BACKEND_VAR,
};
pub use environment::value::{EnvVarError, EnvVarFormatError, EnvVarParseError, TypedEnvVar, TypedEnvVarGuard};
pub use environment::{EnvVar, Environment};

#[cfg(test)]
pub use environment::{guard, instrument, value};
