//! Centralised instrumentation configuration handling.

use std::path::PathBuf;
use std::sync::OnceLock;

use tracing::Level;

use metallic_env::{EnvVar, Environment, InstrumentEnvVar};

/// Errors that can occur while loading or initialising [`AppConfig`].
#[derive(Debug, thiserror::Error)]
pub enum AppConfigError {
    /// The configuration attempted to initialise more than once.
    #[error("app configuration already initialised")]
    AlreadyInitialised,
    /// A provided log level could not be parsed.
    #[error("invalid log level '{value}'")]
    InvalidLogLevel { value: String },
    /// A provided boolean flag could not be parsed.
    #[error("invalid boolean flag '{value}' for {name}")]
    InvalidBoolean { name: &'static str, value: String },
}

/// Global application configuration for instrumentation and logging.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// The minimum tracing level for application logs.
    pub log_level: Level,
    /// Optional path for persisting metrics as JSON lines.
    pub metrics_jsonl_path: Option<PathBuf>,
    /// Whether console metrics should be emitted.
    pub enable_console_metrics: bool,
}

static APP_CONFIG: OnceLock<AppConfig> = OnceLock::new();

impl AppConfig {
    /// Load configuration from the process environment.
    pub fn from_env() -> Result<Self, AppConfigError> {
        let log_level = match Environment::get(InstrumentEnvVar::LogLevel) {
            Some(value) => parse_level(&value)?,
            None => Level::INFO,
        };

        let metrics_jsonl_path = Environment::get(InstrumentEnvVar::MetricsJsonlPath).map(PathBuf::from);

        let console_var = EnvVar::from(InstrumentEnvVar::MetricsConsole);
        let enable_console_metrics = match Environment::get(console_var) {
            Some(value) => parse_bool(console_var, &value)?,
            None => false,
        };

        Ok(Self {
            log_level,
            metrics_jsonl_path,
            enable_console_metrics,
        })
    }

    /// Initialise the global configuration instance from environment variables.
    pub fn initialise_from_env() -> Result<&'static Self, AppConfigError> {
        let config = Self::from_env()?;
        Self::initialise(config)
    }

    /// Store the provided configuration as the global instance.
    pub fn initialise(config: AppConfig) -> Result<&'static Self, AppConfigError> {
        APP_CONFIG.set(config).map_err(|_| AppConfigError::AlreadyInitialised)?;
        Ok(APP_CONFIG.get().expect("configuration just initialised"))
    }

    /// Access the globally-initialised configuration.
    pub fn global() -> &'static Self {
        APP_CONFIG.get().expect("AppConfig not initialised")
    }
}

fn parse_level(value: &str) -> Result<Level, AppConfigError> {
    value
        .parse::<Level>()
        .map_err(|_| AppConfigError::InvalidLogLevel { value: value.to_string() })
}

fn parse_bool(var: EnvVar, value: &str) -> Result<bool, AppConfigError> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AppConfigError::InvalidBoolean {
            name: var.key(),
            value: value.to_string(),
        }),
    }
}
