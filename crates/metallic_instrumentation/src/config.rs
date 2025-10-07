//! Centralised instrumentation configuration handling.

use std::path::PathBuf;
use std::sync::OnceLock;

use tracing::Level;

use metallic_env::EnvVarError;
use metallic_env::environment::instrument::{EMIT_LATENCY, LOG_LEVEL, METRICS_CONSOLE, METRICS_JSONL_PATH};

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
    /// A typed environment variable interaction failed unexpectedly.
    #[error("failed to access instrumentation environment: {source}")]
    EnvVar {
        #[from]
        source: EnvVarError,
    },
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
    /// Whether GPU latency metrics should emit per-command-buffer timings.
    pub emit_latency: bool,
}

static APP_CONFIG: OnceLock<AppConfig> = OnceLock::new();

impl AppConfig {
    /// Load configuration from the process environment.
    pub fn from_env() -> Result<Self, AppConfigError> {
        let log_level = match LOG_LEVEL.get() {
            Ok(Some(value)) => value,
            Ok(None) => Level::INFO,
            Err(EnvVarError::Parse { value, .. }) => return Err(AppConfigError::InvalidLogLevel { value }),
            Err(err) => return Err(err.into()),
        };

        let metrics_jsonl_path = METRICS_JSONL_PATH.get()?;

        let enable_console_metrics = match METRICS_CONSOLE.get() {
            Ok(Some(value)) => value,
            Ok(None) => false,
            Err(EnvVarError::Parse { value, .. }) => {
                return Err(AppConfigError::InvalidBoolean {
                    name: METRICS_CONSOLE.key(),
                    value,
                });
            }
            Err(err) => return Err(err.into()),
        };

        let emit_latency = match EMIT_LATENCY.get() {
            Ok(Some(value)) => value,
            Ok(None) => true,
            Err(EnvVarError::Parse { value, .. }) => {
                return Err(AppConfigError::InvalidBoolean {
                    name: EMIT_LATENCY.key(),
                    value,
                });
            }
            Err(err) => return Err(err.into()),
        };

        Ok(Self {
            log_level,
            metrics_jsonl_path,
            enable_console_metrics,
            emit_latency,
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

    /// Try to access the globally-initialised configuration without panicking.
    pub fn try_global() -> Option<&'static Self> {
        APP_CONFIG.get()
    }
}
