//! Scoped guard helpers for manipulating instrumentation environment variables.

use std::sync::MutexGuard;

use super::{EnvVar, Environment};

#[derive(Debug)]
pub(crate) enum EnvLock<'a> {
    Owned,
    Borrowed(&'a mut MutexGuard<'static, ()>),
}

impl EnvLock<'_> {
    pub(crate) fn with_lock<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MutexGuard<'static, ()>),
    {
        match self {
            EnvLock::Owned => {
                let mut guard = Environment::lock();
                f(&mut guard);
            }
            EnvLock::Borrowed(lock) => {
                f(&mut **lock);
            }
        }
    }
}

/// Guard object that restores the previous environment state upon drop.
///
/// Each mutation acquires the global [`Environment`] mutex so scoped updates
/// remain serialised even when tests execute concurrently. When constructed via
/// the `_with_lock` helpers, the guard borrows the provided [`MutexGuard`]
/// ensuring it cannot outlive the surrounding critical section.
pub struct EnvVarGuard<'a> {
    var: EnvVar,
    previous: Option<String>,
    lock: EnvLock<'a>,
}

impl EnvVarGuard<'static> {
    /// Set the provided environment variable for the duration of the guard.
    pub fn set(var: impl Into<EnvVar>, value: &str) -> Self {
        let var = var.into();
        let mut lock = Environment::lock();
        let previous = Environment::get(var);
        Environment::set_locked(var, value, &mut lock);
        Self {
            var,
            previous,
            lock: EnvLock::Owned,
        }
    }

    /// Unset the provided environment variable for the duration of the guard.
    pub fn unset(var: impl Into<EnvVar>) -> Self {
        let var = var.into();
        let mut lock = Environment::lock();
        let previous = Environment::get(var);
        Environment::remove_locked(var, &mut lock);
        Self {
            var,
            previous,
            lock: EnvLock::Owned,
        }
    }
}

impl<'a> EnvVarGuard<'a> {
    /// Set the provided environment variable while reusing an existing lock.
    ///
    /// The returned guard borrows the supplied lock and must therefore drop
    /// before the mutex guard is released.
    pub fn set_with_lock(var: impl Into<EnvVar>, value: &str, lock: &'a mut MutexGuard<'static, ()>) -> EnvVarGuard<'a> {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::set_locked(var, value, lock);
        EnvVarGuard {
            var,
            previous,
            lock: EnvLock::Borrowed(lock),
        }
    }

    /// Unset the provided environment variable while reusing an existing lock.
    ///
    /// The returned guard borrows the supplied lock and must therefore drop
    /// before the mutex guard is released.
    pub fn unset_with_lock(var: impl Into<EnvVar>, lock: &'a mut MutexGuard<'static, ()>) -> EnvVarGuard<'a> {
        let var = var.into();
        let previous = Environment::get(var);
        Environment::remove_locked(var, lock);
        EnvVarGuard {
            var,
            previous,
            lock: EnvLock::Borrowed(lock),
        }
    }
}

impl<'a> Drop for EnvVarGuard<'a> {
    fn drop(&mut self) {
        self.lock.with_lock(|lock| {
            if let Some(previous) = &self.previous {
                Environment::set_locked(self.var, previous, lock);
            } else {
                Environment::remove_locked(self.var, lock);
            }
        });
    }
}
