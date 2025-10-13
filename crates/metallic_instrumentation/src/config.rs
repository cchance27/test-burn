//! Centralised instrumentation configuration handling.

use std::path::PathBuf;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::Level;

use metallic_env::EnvVarError;
use metallic_env::environment::instrument::{ENABLE_PROFILING, LOG_LEVEL, METRICS_CONSOLE, METRICS_JSONL_PATH};

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
    pub enable_profiling: bool,
}

static APP_CONFIG: OnceLock<AppConfig> = OnceLock::new();
#[cfg(test)]
static RESET_PENDING: AtomicBool = AtomicBool::new(false);
static PROFILING_OVERRIDE_COUNT: AtomicUsize = AtomicUsize::new(0);

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

        let enable_profiling = match ENABLE_PROFILING.get() {
            Ok(Some(value)) => value,
            Ok(None) => true,
            Err(EnvVarError::Parse { value, .. }) => {
                return Err(AppConfigError::InvalidBoolean {
                    name: ENABLE_PROFILING.key(),
                    value,
                });
            }
            Err(err) => return Err(err.into()),
        };

        Ok(Self {
            log_level,
            metrics_jsonl_path,
            enable_console_metrics,
            enable_profiling,
        })
    }

    /// Initialise the global configuration instance from environment variables.
    pub fn initialise_from_env() -> Result<&'static Self, AppConfigError> {
        let config = Self::from_env()?;
        Self::initialise(config)
    }

    /// Store the provided configuration as the global instance.
    pub fn initialise(config: AppConfig) -> Result<&'static Self, AppConfigError> {
        if APP_CONFIG.get().is_some() {
            #[cfg(test)]
            {
                if RESET_PENDING.swap(false, Ordering::SeqCst) {
                    Self::overwrite_for_tests(config);
                    return Ok(APP_CONFIG.get().expect("configuration overwritten after reset"));
                }
            }
            return Err(AppConfigError::AlreadyInitialised);
        }

        APP_CONFIG.set(config).map_err(|_| AppConfigError::AlreadyInitialised)?;
        Ok(APP_CONFIG.get().expect("configuration just initialised"))
    }

    /// Retrieve the global configuration, initialising it from the environment when absent.
    pub fn get_or_init_from_env() -> Result<&'static Self, AppConfigError> {
        if APP_CONFIG.get().is_some() {
            #[cfg(test)]
            {
                let updated = Self::from_env()?;
                Self::overwrite_for_tests(updated);
            }
            return Ok(APP_CONFIG.get().expect("AppConfig already initialised"));
        }

        let config = Self::from_env()?;

        match APP_CONFIG.set(config) {
            Ok(()) => Ok(APP_CONFIG.get().expect("configuration just initialised")),
            Err(_) => Ok(APP_CONFIG.get().expect("configuration concurrently initialised")),
        }
    }

    /// Access the globally-initialised configuration.
    pub fn global() -> &'static Self {
        APP_CONFIG.get().expect("AppConfig not initialised")
    }

    /// Try to access the globally-initialised configuration without panicking.
    pub fn try_global() -> Option<&'static Self> {
        APP_CONFIG.get()
    }

    /// Returns true when profiling has been forced on (typically by tests).
    pub fn profiling_forced() -> bool {
        PROFILING_OVERRIDE_COUNT.load(Ordering::Acquire) > 0
    }

    pub fn force_enable_profiling_guard() -> ProfilingOverrideGuard {
        ProfilingOverrideGuard::enable()
    }

    #[cfg(test)]
    fn default_config() -> Self {
        Self::from_env().unwrap_or_else(|_| Self {
            log_level: Level::INFO,
            metrics_jsonl_path: None,
            enable_console_metrics: false,
            enable_profiling: true,
        })
    }

    #[cfg(test)]
    fn overwrite_for_tests(config: AppConfig) {
        if let Some(existing) = APP_CONFIG.get() {
            unsafe {
                let ptr: *const AppConfig = existing;
                ptr.cast_mut().write(config);
            }
        } else {
            let _ = APP_CONFIG.set(config);
        }
    }

    #[cfg(test)]
    fn ensure_profiling_enabled() {
        if let Some(existing) = APP_CONFIG.get() {
            if !existing.enable_profiling {
                let mut updated = existing.clone();
                updated.enable_profiling = true;
                Self::overwrite_for_tests(updated);
            }
        }
    }
}

pub fn reset_app_config_for_tests() {
    #[cfg(test)]
    {
        RESET_PENDING.store(true, Ordering::SeqCst);
        AppConfig::overwrite_for_tests(AppConfig::default_config());
        PROFILING_OVERRIDE_COUNT.store(0, Ordering::SeqCst);
    }
}

/// RAII guard that forces profiling on for the lifetime of the guard.
pub struct ProfilingOverrideGuard;

impl ProfilingOverrideGuard {
    pub fn enable() -> Self {
        PROFILING_OVERRIDE_COUNT.fetch_add(1, Ordering::AcqRel);
        #[cfg(test)]
        AppConfig::ensure_profiling_enabled();
        Self
    }
}

impl Drop for ProfilingOverrideGuard {
    fn drop(&mut self) {
        let _ = PROFILING_OVERRIDE_COUNT.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |count| count.checked_sub(1));
    }
}
