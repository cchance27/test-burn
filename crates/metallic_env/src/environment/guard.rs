//! Scoped guard helpers for manipulating instrumentation environment variables.

use std::sync::MutexGuard;

use super::{EnvVar, Environment};

/// Guard object that restores the previous environment state upon drop.
///
/// Each mutation acquires the global [`Environment`] mutex so scoped updates
/// remain serialised even when tests execute concurrently.
pub struct EnvVarGuard {
    var: EnvVar,
    previous: Option<String>,
}

impl EnvVarGuard {
    /// Set the provided environment variable for the duration of the guard.
    pub fn set(var: impl Into<EnvVar>, value: &str) -> Self {
        let mut lock = Environment::lock();
        Self::set_with_lock(var, value, &mut lock)
    }

    /// Set the provided environment variable while reusing an existing lock.
    pub fn set_with_lock(var: impl Into<EnvVar>, value: &str, lock: &mut MutexGuard<'static, ()>) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::set_locked(var, value, lock);
        Self { var, previous }
    }

    /// Unset the provided environment variable for the duration of the guard.
    pub fn unset(var: impl Into<EnvVar>) -> Self {
        let mut lock = Environment::lock();
        Self::unset_with_lock(var, &mut lock)
    }

    /// Unset the provided environment variable while reusing an existing lock.
    pub fn unset_with_lock(var: impl Into<EnvVar>, lock: &mut MutexGuard<'static, ()>) -> Self {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::remove_locked(var, lock);
        Self { var, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        let mut lock = Environment::lock();
        if let Some(previous) = &self.previous {
            Environment::set_locked(self.var, previous, &mut lock);
        } else {
            Environment::remove_locked(self.var, &mut lock);
        }
    }
}
