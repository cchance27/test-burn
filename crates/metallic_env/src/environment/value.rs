//! Typed environment variable descriptors and guard helpers.
//!
//! This module introduces [`TypedEnvVar`], a thin wrapper around [`EnvVar`]
//! that performs parsing and formatting using caller-supplied callbacks.
//! The descriptor surfaces ergonomic setters, getters, and scoped mutation
//! guards that automatically restore the previous process environment state.
//!
//! # Examples
//!
//! ```
//! use metallic_env::environment::instrument::LOG_LEVEL;
//! use tracing::Level;
//!
//! let _lock = metallic_env::Environment::lock();
//! let _guard = LOG_LEVEL.set_guard(Level::DEBUG).expect("set log level");
//! assert_eq!(*_guard, Level::DEBUG);
//! ```

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::MutexGuard;

use super::guard::{EnvLock, EnvVarGuard};
use super::{EnvVar, Environment};

/// Errors emitted when interacting with typed environment variables.
#[derive(Debug, thiserror::Error)]
pub enum EnvVarError {
    /// The environment value could not be parsed into the desired type.
    #[error("failed to parse environment variable {name} from '{value}': {source}")]
    Parse {
        /// The canonical environment variable name.
        name: &'static str,
        /// The raw value retrieved from the process environment.
        value: String,
        /// The underlying parse error.
        source: EnvVarParseError,
    },
    /// The provided value could not be formatted for storage.
    #[error("failed to format environment variable {name}: {source}")]
    Format {
        /// The canonical environment variable name.
        name: &'static str,
        /// The underlying formatting error.
        source: EnvVarFormatError,
    },
}

/// Error produced by a [`TypedEnvVar`] parsing callback.
#[derive(Debug, Clone)]
pub struct EnvVarParseError {
    message: String,
}

impl EnvVarParseError {
    /// Construct a new parse error with the provided message.
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for EnvVarParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EnvVarParseError {}

impl From<&'static str> for EnvVarParseError {
    fn from(value: &'static str) -> Self {
        Self {
            message: value.to_string(),
        }
    }
}

impl From<String> for EnvVarParseError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

/// Error produced by a [`TypedEnvVar`] formatting callback.
#[derive(Debug, Clone)]
pub struct EnvVarFormatError {
    message: String,
}

impl EnvVarFormatError {
    /// Construct a new format error with the provided message.
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for EnvVarFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EnvVarFormatError {}

impl From<&'static str> for EnvVarFormatError {
    fn from(value: &'static str) -> Self {
        Self {
            message: value.to_string(),
        }
    }
}

impl From<String> for EnvVarFormatError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

/// Callback used to parse an environment string into a concrete value.
pub type ParseFn<T> = fn(&str) -> Result<T, EnvVarParseError>;
/// Callback used to format a typed value before storing it in the environment.
pub type FormatFn<T> = fn(&T) -> Result<String, EnvVarFormatError>;

/// Descriptor for a strongly-typed environment variable.
#[derive(Clone, Copy)]
pub struct TypedEnvVar<T> {
    var: EnvVar,
    parse: ParseFn<T>,
    format: FormatFn<T>,
    _marker: PhantomData<T>,
}

impl<T> TypedEnvVar<T> {
    fn format_value(&self, value: &T) -> Result<String, EnvVarError> {
        (self.format)(value).map_err(|source| EnvVarError::Format { name: self.key(), source })
    }

    /// Create a new typed descriptor using the provided callbacks.
    pub const fn new(var: EnvVar, parse: ParseFn<T>, format: FormatFn<T>) -> Self {
        Self {
            var,
            parse,
            format,
            _marker: PhantomData,
        }
    }

    /// Retrieve the canonical environment variable key.
    pub const fn key(&self) -> &'static str {
        self.var.key()
    }

    /// Retrieve the underlying [`EnvVar`] identifier.
    pub const fn var(&self) -> EnvVar {
        self.var
    }

    /// Read the environment variable and parse it into the typed value.
    pub fn get(&self) -> Result<Option<T>, EnvVarError> {
        match Environment::get(self.var) {
            Some(raw) => (self.parse)(&raw).map(Some).map_err(|source| EnvVarError::Parse {
                name: self.key(),
                value: raw,
                source,
            }),
            None => Ok(None),
        }
    }

    /// Set the environment variable to the provided typed value.
    pub fn set(&self, value: T) -> Result<(), EnvVarError> {
        let formatted = self.format_value(&value)?;
        Environment::set(self.var, &formatted);
        Ok(())
    }

    /// Remove the environment variable from the process environment.
    pub fn unset(&self) {
        Environment::remove(self.var);
    }

    /// Set the environment variable for the lifetime of the returned guard.
    pub fn set_guard(&self, value: T) -> Result<TypedEnvVarGuard<'_, T>, EnvVarError> {
        let formatted = self.format_value(&value)?;
        let mut lock = Environment::lock();
        let previous = Environment::get(self.var);
        Environment::set_locked(self.var, &formatted, &mut lock);
        Ok(TypedEnvVarGuard {
            descriptor: self,
            previous,
            value: Some(value),
            lock: EnvLock::Owned,
        })
    }

    /// Set the environment variable for the guard lifetime while reusing a lock.
    pub fn set_guard_with_lock<'a>(
        &'a self,
        value: T,
        lock: &'a mut MutexGuard<'static, ()>,
    ) -> Result<TypedEnvVarGuard<'a, T>, EnvVarError> {
        let formatted = self.format_value(&value)?;
        let previous = Environment::get(self.var);
        Environment::set_locked(self.var, &formatted, lock);
        Ok(TypedEnvVarGuard {
            descriptor: self,
            previous,
            value: Some(value),
            lock: EnvLock::Borrowed(lock),
        })
    }

    /// Unset the environment variable for the lifetime of the guard.
    pub fn unset_guard(&self) -> EnvVarGuard<'_> {
        EnvVarGuard::unset(self.var)
    }

    /// Unset the environment variable for the guard lifetime while reusing a lock.
    pub fn unset_guard_with_lock<'a>(&self, lock: &'a mut MutexGuard<'static, ()>) -> EnvVarGuard<'a> {
        EnvVarGuard::unset_with_lock(self.var, lock)
    }
}

/// Guard that restores the previous state of a typed environment variable.
///
/// Dropping the guard acquires the global environment mutex before restoring
/// the prior value to guarantee serialised mutations. When created through the
/// `_with_lock` helpers it borrows the caller's mutex guard, ensuring the guard
/// cannot outlive the critical section that produced it.
pub struct TypedEnvVarGuard<'a, T> {
    descriptor: &'a TypedEnvVar<T>,
    previous: Option<String>,
    value: Option<T>,
    lock: EnvLock<'a>,
}

impl<'a, T> TypedEnvVarGuard<'a, T> {
    /// Consume the guard, returning the typed value that was set.
    pub fn into_inner(mut self) -> T {
        self.value.take().expect("typed environment guard should always contain a value")
    }
}

impl<'a, T> Deref for TypedEnvVarGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("typed environment guard should always contain a value")
    }
}

impl<'a, T> Drop for TypedEnvVarGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.with_lock(|lock| {
            if let Some(previous) = &self.previous {
                Environment::set_locked(self.descriptor.var, previous, lock);
            } else {
                Environment::remove_locked(self.descriptor.var, lock);
            }
        });
    }
}
