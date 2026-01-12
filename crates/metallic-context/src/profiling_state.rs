use std::sync::atomic::{AtomicBool, Ordering};

use metallic_env::environment::instrument::ENABLE_PROFILING_VAR;

static GLOBAL_PROFILING_STATE: AtomicBool = AtomicBool::new(false);

pub fn initialize_profiling_state_from_env() {
    let initial_state = match ENABLE_PROFILING_VAR.get() {
        Ok(Some(value)) => value,
        Ok(None) => false, // Default to false when not set for performance
        Err(_) => false,   // Default to false if there's an error
    };

    GLOBAL_PROFILING_STATE.store(initial_state, Ordering::Relaxed);
}

pub fn get_profiling_state() -> bool {
    GLOBAL_PROFILING_STATE.load(Ordering::Relaxed)
}

pub fn set_profiling_state(enabled: bool) {
    GLOBAL_PROFILING_STATE.store(enabled, Ordering::Relaxed);
}

pub fn toggle_profiling_state() -> bool {
    GLOBAL_PROFILING_STATE.fetch_xor(true, Ordering::Relaxed)
}
