//! Scoped guard helpers for manipulating instrumentation environment variables.

use super::{EnvVar, Environment};

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
