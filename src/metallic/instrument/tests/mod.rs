#![cfg(test)]

use std::sync::{Mutex, OnceLock};

mod config;
mod exporters;
mod layer;

fn set_env_var(key: &str, value: &str) {
    // Safety: the standard library marks these interfaces as unsafe for our
    // target configuration, but the inputs we provide are constant ASCII
    // strings, ensuring they are well-formed and null-terminated as required.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env_var(key: &str) {
    // Safety: removing a variable with a constant ASCII key is always valid
    // for the platform configuration used in tests.
    unsafe {
        std::env::remove_var(key);
    }
}

pub(super) fn env_mutex() -> &'static Mutex<()> {
    static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_MUTEX.get_or_init(|| Mutex::new(()))
}

pub(super) struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    pub(super) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        set_env_var(key, value);
        Self { key, previous }
    }

    pub(super) fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        remove_env_var(key);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            set_env_var(self.key, previous);
        } else {
            remove_env_var(self.key);
        }
    }
}
