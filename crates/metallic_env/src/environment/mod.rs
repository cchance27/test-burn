//! Process environment abstractions shared across Metallic instrumentation components.

pub mod foundry;
pub mod guard;
pub mod instrument;
pub mod value;

use std::{
    collections::HashMap, sync::{Mutex, MutexGuard, OnceLock}
};

use foundry::FoundryEnvVar;
use instrument::InstrumentEnvVar;

/// Namespaced environment variable identifiers used by instrumentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnvVar {
    /// Environment variables specific to foundry/runtime execution.
    Foundry(FoundryEnvVar),
    /// Environment variables specific to the instrumentation pipeline.
    Instrument(InstrumentEnvVar),
}

impl From<FoundryEnvVar> for EnvVar {
    fn from(value: FoundryEnvVar) -> Self {
        Self::Foundry(value)
    }
}

impl From<InstrumentEnvVar> for EnvVar {
    fn from(value: InstrumentEnvVar) -> Self {
        Self::Instrument(value)
    }
}

impl EnvVar {
    /// Retrieve the canonical environment variable key for the identifier.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            EnvVar::Foundry(inner) => inner.key(),
            EnvVar::Instrument(inner) => inner.key(),
        }
    }
}

/// Process environment facade that centralises access and synchronisation.
pub struct Environment;

/// Guard that pops one override layer when dropped.
pub struct EnvironmentOverrideGuard {
    active: bool,
}

impl Drop for EnvironmentOverrideGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(stack) = ENV_OVERRIDE_STACK.get()
            && let Ok(mut stack) = stack.lock()
        {
            stack.pop();
        }
    }
}

type OverrideMap = HashMap<String, String>;

static ENV_OVERRIDE_STACK: OnceLock<Mutex<Vec<OverrideMap>>> = OnceLock::new();

impl Environment {
    /// Acquire the global environment mutex, ensuring serialised mutations.
    pub fn lock() -> MutexGuard<'static, ()> {
        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_MUTEX.get_or_init(|| Mutex::new(())).lock().expect("environment mutex poisoned")
    }

    /// Push a process-scoped override layer used by [`Environment::get`].
    ///
    /// Lookups consult the most recent override first, then fall back to the
    /// process environment. Drop the returned guard to restore the previous
    /// override stack.
    pub fn push_overrides(overrides: OverrideMap) -> EnvironmentOverrideGuard {
        let stack = ENV_OVERRIDE_STACK.get_or_init(|| Mutex::new(Vec::new()));
        stack.lock().expect("environment override stack poisoned").push(overrides);
        EnvironmentOverrideGuard { active: true }
    }

    fn get_override(key: &str) -> Option<String> {
        let stack = ENV_OVERRIDE_STACK.get()?;
        let stack = stack.lock().ok()?;
        stack.iter().rev().find_map(|layer| layer.get(key).cloned())
    }

    /// Read the environment variable as a UTF-8 string if present.
    pub fn get(var: impl Into<EnvVar>) -> Option<String> {
        let var = var.into();
        if let Some(override_value) = Self::get_override(var.key()) {
            return Some(override_value);
        }
        let _guard = Self::lock();
        std::env::var(var.key()).ok()
    }

    /// Set the environment variable using the provided UTF-8 value.
    ///
    /// This method acquires and releases the global environment mutex for the
    /// duration of the call so isolated mutations remain serialised without the
    /// caller managing synchronisation manually.
    pub fn set(var: impl Into<EnvVar>, value: &str) {
        let var = var.into();
        let _guard = Self::lock();
        // SAFETY: Holding the mutex guard serialises environment access across
        // threads, satisfying the requirements of `std::env::set_var`.
        unsafe { std::env::set_var(var.key(), value) };
    }

    /// Remove the environment variable from the process environment.
    ///
    /// Like [`Environment::set`], this acquires and releases the global mutex for
    /// the duration of the operation.
    pub fn remove(var: impl Into<EnvVar>) {
        let var = var.into();
        let _guard = Self::lock();
        // SAFETY: The mutex guard serialises access across threads, satisfying
        // the requirements of `std::env::remove_var`.
        unsafe { std::env::remove_var(var.key()) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::{Environment, FoundryEnvVar};

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        TEST_MUTEX.get_or_init(|| Mutex::new(())).lock().expect("test mutex poisoned")
    }

    #[test]
    fn override_layer_precedes_process_environment() {
        let _lock = test_lock();
        Environment::set(FoundryEnvVar::AccumDtype, "f32");
        let mut overrides = std::collections::HashMap::new();
        overrides.insert("METALLIC_ACCUM_DTYPE".to_string(), "bf16".to_string());
        let _guard = Environment::push_overrides(overrides);
        assert_eq!(Environment::get(FoundryEnvVar::AccumDtype).as_deref(), Some("bf16"));
        Environment::remove(FoundryEnvVar::AccumDtype);
    }

    #[test]
    fn override_layer_is_restored_after_guard_drop() {
        let _lock = test_lock();
        Environment::set(FoundryEnvVar::AccumDtype, "f32");
        {
            let mut overrides = std::collections::HashMap::new();
            overrides.insert("METALLIC_ACCUM_DTYPE".to_string(), "f16".to_string());
            let _guard = Environment::push_overrides(overrides);
            assert_eq!(Environment::get(FoundryEnvVar::AccumDtype).as_deref(), Some("f16"));
        }
        assert_eq!(Environment::get(FoundryEnvVar::AccumDtype).as_deref(), Some("f32"));
        Environment::remove(FoundryEnvVar::AccumDtype);
    }
}
