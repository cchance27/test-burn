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

/// Typed descriptor for the instrumentation log level.
pub const LOG_LEVEL: TypedEnvVar<Level> = TypedEnvVar::new(InstrumentEnvVar::LogLevel.into_env(), parse_log_level, format_level);

/// Typed descriptor for the metrics JSONL output path.
pub const METRICS_JSONL_PATH: TypedEnvVar<PathBuf> =
    TypedEnvVar::new(InstrumentEnvVar::MetricsJsonlPath.into_env(), parse_path, format_path);

/// Typed descriptor for the console metrics toggle.
pub const METRICS_CONSOLE: TypedEnvVar<bool> = TypedEnvVar::new(InstrumentEnvVar::MetricsConsole.into_env(), parse_bool, format_bool);

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
    pub fn unset_guard(&self) -> EnvVarGuard {
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
    pub fn unset_guard(&self) -> EnvVarGuard {
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
    pub fn unset_guard(&self) -> EnvVarGuard {
        METRICS_CONSOLE.unset_guard()
    }
}

/// Ergonomic constant mirroring the previous enum-style API for the log level.
pub const LogLevel: InstrumentLogLevel = InstrumentLogLevel;
/// Ergonomic constant mirroring the previous enum-style API for the JSONL path.
pub const MetricsJsonlPath: InstrumentMetricsJsonlPath = InstrumentMetricsJsonlPath;
/// Ergonomic constant mirroring the previous enum-style API for the console toggle.
pub const MetricsConsole: InstrumentMetricsConsole = InstrumentMetricsConsole;

fn parse_log_level(value: &str) -> Result<Level, EnvVarParseError> {
    value.parse::<Level>().map_err(|_| EnvVarParseError::new("invalid tracing level"))
}

fn format_level(level: &Level) -> Result<String, EnvVarFormatError> {
    Ok(level.to_string())
}

fn parse_path(value: &str) -> Result<PathBuf, EnvVarParseError> {
    Ok(PathBuf::from(value))
}

fn format_path(path: &PathBuf) -> Result<String, EnvVarFormatError> {
    Ok(path.to_string_lossy().into_owned())
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
