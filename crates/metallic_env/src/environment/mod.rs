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
    ///
    /// This method acquires the global environment mutex, ensuring callers do not
    /// have to manage synchronisation for isolated mutations. Prefer
    /// [`Environment::lock`] when batching multiple operations so they can share a
    /// single critical section.
    pub fn set(var: impl Into<EnvVar>, value: &str) {
        let var = var.into();
        let mut guard = Self::lock();
        Self::set_locked(var, value, &mut guard);
    }

    /// Remove the environment variable from the process environment.
    ///
    /// Like [`Environment::set`], this acquires the global mutex internally. Hold
    /// the lock manually via [`Environment::lock`] if you need to couple the
    /// deletion with additional reads or writes.
    pub fn remove(var: impl Into<EnvVar>) {
        let var = var.into();
        let mut guard = Self::lock();
        Self::remove_locked(var, &mut guard);
    }

    /// Set the environment variable while reusing an existing environment lock.
    ///
    /// # Safety
    ///
    /// `std::env::set_var` is an `unsafe` API because concurrent mutation of the
    /// process environment is undefined behaviour. Holding the guard ensures the
    /// caller has serialised access to the environment for the duration of the
    /// call.
    pub(crate) fn set_locked(var: EnvVar, value: &str, _guard: &mut MutexGuard<'static, ()>) {
        // SAFETY: The guard parameter proves the caller currently holds the
        // global environment mutex, preventing concurrent mutation.
        unsafe { std::env::set_var(var.key(), value) };
    }

    /// Remove the environment variable while reusing an existing environment lock.
    ///
    /// # Safety
    ///
    /// `std::env::remove_var` has the same safety requirements as
    /// [`std::env::set_var`]; the environment mutex guard ensures no other
    /// thread can mutate the process environment concurrently.
    pub(crate) fn remove_locked(var: EnvVar, _guard: &mut MutexGuard<'static, ()>) {
        // SAFETY: Serialisation is enforced by the guard parameter which holds
        // the global environment mutex for the lifetime of this call.
        unsafe { std::env::remove_var(var.key()) };
    }
}
