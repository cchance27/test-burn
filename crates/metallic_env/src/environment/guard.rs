//! Scoped guard helpers for manipulating instrumentation environment variables.

use std::marker::PhantomData;

use super::{EnvVar, Environment};

/// Guard object that restores the previous environment state upon drop.
///
/// Each mutation acquires and releases the global [`Environment`] mutex so
/// scoped updates remain serialised even when tests execute concurrently. The
/// guard does not hold the mutex for its entire lifetime, which prevents
/// deadlocks when combined with other helpers that perform their own locking.
pub struct EnvVarGuard<'a> {
    var: EnvVar,
    previous: Option<String>,
    _marker: PhantomData<&'a ()>,
}

impl EnvVarGuard<'static> {
    /// Set the provided environment variable for the duration of the guard.
    pub fn set(var: impl Into<EnvVar>, value: &str) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::set(var, value);
        Self {
            var,
            previous,
            _marker: PhantomData,
        }
    }

    /// Unset the provided environment variable for the duration of the guard.
    pub fn unset(var: impl Into<EnvVar>) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::remove(var);
        Self {
            var,
            previous,
            _marker: PhantomData,
        }
    }
}

impl<'a> Drop for EnvVarGuard<'a> {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            Environment::set(self.var, previous);
        } else {
            Environment::remove(self.var);
        }
    }
}
