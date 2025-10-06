//! Process environment helpers for instrumentation configuration.

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Namespaced environment variable identifiers used by instrumentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnvVar {
    /// Environment variables specific to the instrumentation pipeline.
    Instrument(InstrumentEnvVar),
}

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

impl From<InstrumentEnvVar> for EnvVar {
    fn from(value: InstrumentEnvVar) -> Self {
        Self::Instrument(value)
    }
}

impl EnvVar {
    /// Retrieve the canonical environment variable key for the identifier.
    pub const fn key(self) -> &'static str {
        match self {
            EnvVar::Instrument(inner) => inner.key(),
        }
    }
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

/// Guard object that restores the previous environment state upon drop.
pub struct EnvVarGuard {
    var: EnvVar,
    previous: Option<String>,
}

impl EnvVarGuard {
    /// Set the provided environment variable for the duration of the guard.
    pub fn set(var: impl Into<EnvVar>, value: &str) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::set(var, value);
        Self { var, previous }
    }

    /// Unset the provided environment variable for the duration of the guard.
    pub fn unset(var: impl Into<EnvVar>) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::remove(var);
        Self { var, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            Environment::set(self.var, previous);
        } else {
            Environment::remove(self.var);
        }
    }
}

/// Process environment facade that centralises access and synchronisation.
pub struct Environment;

impl Environment {
    /// Acquire the global environment mutex, ensuring serialised mutations.
    pub fn lock() -> MutexGuard<'static, ()> {
        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_MUTEX.get_or_init(|| Mutex::new(())).lock().expect("environment mutex poisoned")
    }

    /// Read the environment variable as a UTF-8 string if present.
    pub fn get(var: impl Into<EnvVar>) -> Option<String> {
        let var = var.into();
        std::env::var(var.key()).ok()
    }

    /// Set the environment variable using the provided UTF-8 value.
    #[allow(unused_unsafe)]
    pub fn set(var: impl Into<EnvVar>, value: &str) {
        let var = var.into();
        // SAFETY: Environment mutations are synchronised via [`Environment::lock`].
        unsafe {
            std::env::set_var(var.key(), value);
        }
    }

    /// Remove the environment variable from the process environment.
    #[allow(unused_unsafe)]
    pub fn remove(var: impl Into<EnvVar>) {
        let var = var.into();
        // SAFETY: Environment mutations are synchronised via [`Environment::lock`].
        unsafe {
            std::env::remove_var(var.key());
        }
    }
}
