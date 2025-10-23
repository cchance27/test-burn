//! Shared process environment helpers for Metallic instrumentation.

pub mod environment;

pub use environment::{
    EnvVar, Environment, guard::EnvVarGuard, instrument::{
        ENABLE_PROFILING, ENABLE_PROFILING_VAR, FORCE_MATMUL_BACKEND, FORCE_MATMUL_BACKEND_VAR, FORCE_SDPA_BACKEND, FORCE_SDPA_BACKEND_VAR, InstrumentEnableProfiling, InstrumentEnvVar, InstrumentForceMatmulBackend, InstrumentForceSdpaBackend, InstrumentLogLevel, InstrumentMatmulSimdMMin, InstrumentMatmulSimdNMin, InstrumentMatmulSmallnMaxN, InstrumentMetricsConsole, InstrumentMetricsJsonlPath, InstrumentSoftmaxBackend, LOG_LEVEL, LOG_LEVEL_VAR, MATMUL_SIMD_M_MIN, MATMUL_SIMD_M_MIN_VAR, MATMUL_SIMD_N_MIN, MATMUL_SIMD_N_MIN_VAR, MATMUL_SMALLN_MAX_N, MATMUL_SMALLN_MAX_N_VAR, METRICS_CONSOLE, METRICS_CONSOLE_VAR, METRICS_JSONL_PATH, METRICS_JSONL_PATH_VAR, SOFTMAX_BACKEND, SOFTMAX_BACKEND_VAR
    }, value::{EnvVarError, EnvVarFormatError, EnvVarParseError, TypedEnvVar, TypedEnvVarGuard}
};
#[cfg(test)]
pub use environment::{guard, instrument, value};
