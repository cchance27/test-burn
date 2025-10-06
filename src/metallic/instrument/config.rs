//! Centralised instrumentation configuration handling.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

use tracing::Level;

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
        let log_level = match env::var("METALLIC_LOG_LEVEL") {
            Ok(value) => parse_level(&value)?,
            Err(_) => Level::INFO,
        };

        let metrics_jsonl_path = env::var("METALLIC_METRICS_JSONL_PATH").ok().map(PathBuf::from);

        let enable_console_metrics = match env::var("METALLIC_METRICS_CONSOLE") {
            Ok(value) => parse_bool("METALLIC_METRICS_CONSOLE", &value)?,
            Err(_) => false,
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

fn parse_bool(name: &'static str, value: &str) -> Result<bool, AppConfigError> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AppConfigError::InvalidBoolean {
            name,
            value: value.to_string(),
        }),
    }
}
