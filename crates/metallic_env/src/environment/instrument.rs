//! Instrumentation-specific environment variable identifiers and descriptors.

use std::path::PathBuf;
use tracing::Level;

use super::EnvVar;
use super::guard::EnvVarGuard;
use super::value::{EnvVarError, EnvVarFormatError, EnvVarParseError, TypedEnvVar, TypedEnvVarGuard};

/// Instrumentation-specific environment variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InstrumentEnvVar {
    /// Controls the global log level for the instrumentation pipeline.
    LogLevel,
    /// Path where metrics should be persisted as JSONL.
    MetricsJsonlPath,
    /// Enables console metrics emission when set to a truthy value.
    MetricsConsole,
    /// Enables per-command-buffer GPU latency emission and more detailed profiling instrumentation (costly to performance.)
    EnableProfiling,
    /// Forces the matmul backend to a specific implementation for testing.
    ForceMatmulBackend,
    /// Forces the softmax backend to a specific implementation for testing.
    SoftmaxBackend,
}

impl InstrumentEnvVar {
    /// Obtain the canonical environment variable key for the identifier.
    pub const fn key(self) -> &'static str {
        match self {
            InstrumentEnvVar::LogLevel => "METALLIC_LOG_LEVEL",
            InstrumentEnvVar::MetricsJsonlPath => "METALLIC_METRICS_JSONL_PATH",
            InstrumentEnvVar::MetricsConsole => "METALLIC_METRICS_CONSOLE",
            InstrumentEnvVar::EnableProfiling => "METALLIC_ENABLE_PROFILING",
            InstrumentEnvVar::ForceMatmulBackend => "METALLIC_FORCE_MATMUL_BACKEND",
            InstrumentEnvVar::SoftmaxBackend => "METALLIC_SOFTMAX_BACKEND",
        }
    }

    /// Convert into the unscoped [`EnvVar`] variant.
    pub const fn into_env(self) -> EnvVar {
        EnvVar::Instrument(self)
    }
}

/// Typed descriptor for the instrumentation log level.
pub const LOG_LEVEL: TypedEnvVar<Level> = TypedEnvVar::new(InstrumentEnvVar::LogLevel.into_env(), parse_log_level, format_level);

/// Typed descriptor for the metrics JSONL output path.
pub const METRICS_JSONL_PATH: TypedEnvVar<PathBuf> =
    TypedEnvVar::new(InstrumentEnvVar::MetricsJsonlPath.into_env(), parse_path, format_path);

/// Typed descriptor for the console metrics toggle.
pub const METRICS_CONSOLE: TypedEnvVar<bool> = TypedEnvVar::new(InstrumentEnvVar::MetricsConsole.into_env(), parse_bool, format_bool);

/// Typed descriptor for the per-command-buffer latency emission toggle.
pub const ENABLE_PROFILING: TypedEnvVar<bool> = TypedEnvVar::new(InstrumentEnvVar::EnableProfiling.into_env(), parse_bool, format_bool);

/// Typed descriptor for forcing the matmul backend.
pub const FORCE_MATMUL_BACKEND: TypedEnvVar<String> =
    TypedEnvVar::new(InstrumentEnvVar::ForceMatmulBackend.into_env(), parse_string, format_string);

/// Typed descriptor for forcing the softmax backend.
pub const SOFTMAX_BACKEND: TypedEnvVar<String> =
    TypedEnvVar::new(InstrumentEnvVar::SoftmaxBackend.into_env(), parse_string, format_string);

/// Shim exposing ergonomic helpers for the log level variable.
pub struct InstrumentLogLevel;

impl InstrumentLogLevel {
    /// Retrieve the descriptor associated with the log level variable.
    pub const fn descriptor(&self) -> TypedEnvVar<Level> {
        LOG_LEVEL
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        LOG_LEVEL.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<Level>, EnvVarError> {
        LOG_LEVEL.get()
    }

    /// Set the environment variable to the provided level.
    pub fn set(&self, value: Level) -> Result<(), EnvVarError> {
        LOG_LEVEL.set(value)
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: Level) -> Result<TypedEnvVarGuard<'_, Level>, EnvVarError> {
        LOG_LEVEL.set_guard(value)
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        LOG_LEVEL.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        LOG_LEVEL.unset_guard()
    }
}

/// Shim exposing ergonomic helpers for the metrics JSONL path variable.
pub struct InstrumentMetricsJsonlPath;

impl InstrumentMetricsJsonlPath {
    /// Retrieve the descriptor associated with the JSONL path variable.
    pub const fn descriptor(&self) -> TypedEnvVar<PathBuf> {
        METRICS_JSONL_PATH
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        METRICS_JSONL_PATH.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<PathBuf>, EnvVarError> {
        METRICS_JSONL_PATH.get()
    }

    /// Set the environment variable to the provided path.
    pub fn set(&self, value: impl Into<PathBuf>) -> Result<(), EnvVarError> {
        METRICS_JSONL_PATH.set(value.into())
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: impl Into<PathBuf>) -> Result<TypedEnvVarGuard<'_, PathBuf>, EnvVarError> {
        METRICS_JSONL_PATH.set_guard(value.into())
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        METRICS_JSONL_PATH.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        METRICS_JSONL_PATH.unset_guard()
    }
}

/// Shim exposing ergonomic helpers for the console metrics toggle.
pub struct InstrumentMetricsConsole;

impl InstrumentMetricsConsole {
    /// Retrieve the descriptor associated with the console metrics toggle.
    pub const fn descriptor(&self) -> TypedEnvVar<bool> {
        METRICS_CONSOLE
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        METRICS_CONSOLE.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<bool>, EnvVarError> {
        METRICS_CONSOLE.get()
    }

    /// Set the environment variable to the provided toggle state.
    pub fn set(&self, value: bool) -> Result<(), EnvVarError> {
        METRICS_CONSOLE.set(value)
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: bool) -> Result<TypedEnvVarGuard<'_, bool>, EnvVarError> {
        METRICS_CONSOLE.set_guard(value)
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        METRICS_CONSOLE.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        METRICS_CONSOLE.unset_guard()
    }
}

/// Shim exposing ergonomic helpers for the per-command-buffer latency toggle.
pub struct InstrumentEnableProfiling;

impl InstrumentEnableProfiling {
    /// Retrieve the descriptor associated with the latency toggle.
    pub const fn descriptor(&self) -> TypedEnvVar<bool> {
        ENABLE_PROFILING
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        ENABLE_PROFILING.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<bool>, EnvVarError> {
        ENABLE_PROFILING.get()
    }

    /// Set the environment variable to the provided toggle state.
    pub fn set(&self, value: bool) -> Result<(), EnvVarError> {
        ENABLE_PROFILING.set(value)
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: bool) -> Result<TypedEnvVarGuard<'_, bool>, EnvVarError> {
        ENABLE_PROFILING.set_guard(value)
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        ENABLE_PROFILING.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        ENABLE_PROFILING.unset_guard()
    }
}

/// Shim exposing ergonomic helpers for the matmul backend override.
pub struct InstrumentForceMatmulBackend;

impl InstrumentForceMatmulBackend {
    /// Retrieve the descriptor associated with the matmul backend override.
    pub const fn descriptor(&self) -> TypedEnvVar<String> {
        FORCE_MATMUL_BACKEND
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        FORCE_MATMUL_BACKEND.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<String>, EnvVarError> {
        FORCE_MATMUL_BACKEND.get()
    }

    /// Set the environment variable to the provided backend.
    pub fn set(&self, value: impl Into<String>) -> Result<(), EnvVarError> {
        FORCE_MATMUL_BACKEND.set(value.into())
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: impl Into<String>) -> Result<TypedEnvVarGuard<'_, String>, EnvVarError> {
        FORCE_MATMUL_BACKEND.set_guard(value.into())
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        FORCE_MATMUL_BACKEND.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        FORCE_MATMUL_BACKEND.unset_guard()
    }
}

/// Shim exposing ergonomic helpers for the softmax backend override.
pub struct InstrumentSoftmaxBackend;

impl InstrumentSoftmaxBackend {
    /// Retrieve the descriptor associated with the softmax backend override.
    pub const fn descriptor(&self) -> TypedEnvVar<String> {
        SOFTMAX_BACKEND
    }

    /// Canonical key for the environment variable.
    pub const fn key(&self) -> &'static str {
        SOFTMAX_BACKEND.key()
    }

    /// Retrieve the typed value, if set.
    pub fn get(&self) -> Result<Option<String>, EnvVarError> {
        SOFTMAX_BACKEND.get()
    }

    /// Set the environment variable to the provided backend.
    pub fn set(&self, value: impl Into<String>) -> Result<(), EnvVarError> {
        SOFTMAX_BACKEND.set(value.into())
    }

    /// Set the environment variable for the guard's lifetime.
    pub fn set_guard(&self, value: impl Into<String>) -> Result<TypedEnvVarGuard<'_, String>, EnvVarError> {
        SOFTMAX_BACKEND.set_guard(value.into())
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        SOFTMAX_BACKEND.unset()
    }

    /// Unset the environment variable for the guard's lifetime.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        SOFTMAX_BACKEND.unset_guard()
    }
}

/// Ergonomic constant exposing the log-level helper methods.
pub const LOG_LEVEL_VAR: InstrumentLogLevel = InstrumentLogLevel;
/// Ergonomic constant exposing the JSONL-path helper methods.
pub const METRICS_JSONL_PATH_VAR: InstrumentMetricsJsonlPath = InstrumentMetricsJsonlPath;
/// Ergonomic constant exposing the console-toggle helper methods.
pub const METRICS_CONSOLE_VAR: InstrumentMetricsConsole = InstrumentMetricsConsole;
/// Ergonomic constant exposing the latency-toggle helper methods.
pub const ENABLE_PROFILING_VAR: InstrumentEnableProfiling = InstrumentEnableProfiling;

/// Ergonomic constant exposing the matmul-backend-override helper methods.
pub const FORCE_MATMUL_BACKEND_VAR: InstrumentForceMatmulBackend = InstrumentForceMatmulBackend;

/// Ergonomic constant exposing the softmax-backend-override helper methods.
pub const SOFTMAX_BACKEND_VAR: InstrumentSoftmaxBackend = InstrumentSoftmaxBackend;

fn parse_log_level(value: &str) -> Result<Level, EnvVarParseError> {
    value.parse::<Level>().map_err(|_| EnvVarParseError::new("invalid tracing level"))
}

fn format_level(level: &Level) -> Result<String, EnvVarFormatError> {
    Ok(level.to_string())
}

fn parse_path(value: &str) -> Result<PathBuf, EnvVarParseError> {
    Ok(PathBuf::from(value))
}

fn format_path(path: &impl AsRef<std::path::Path>) -> Result<String, EnvVarFormatError> {
    Ok(path.as_ref().to_string_lossy().into_owned())
}

fn parse_string(value: &str) -> Result<String, EnvVarParseError> {
    Ok(value.to_string())
}

fn format_string(value: &String) -> Result<String, EnvVarFormatError> {
    Ok(value.clone())
}

fn parse_bool(value: &str) -> Result<bool, EnvVarParseError> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(EnvVarParseError::new("value is not a recognised boolean")),
    }
}

fn format_bool(value: &bool) -> Result<String, EnvVarFormatError> {
    Ok(value.to_string())
}
