//! Process environment abstractions shared across Metallic instrumentation components.

pub mod guard;
pub mod instrument;
pub mod value;

use std::sync::{Mutex, MutexGuard, OnceLock};

use instrument::InstrumentEnvVar;

/// Namespaced environment variable identifiers used by instrumentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnvVar {
    /// Environment variables specific to the instrumentation pipeline.
    Instrument(InstrumentEnvVar),
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
