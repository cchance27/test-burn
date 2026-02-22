use std::sync::{
    OnceLock, atomic::{AtomicBool, AtomicU64, Ordering}
};

use metallic_env::{FOUNDRY_PER_KERNEL_PROFILING, FoundryEnvVar, RECORD_CB_GPU_TIMING, is_set};
use rustc_hash::FxHashMap;

// Real-time profiling state for Foundry (toggled by Ctrl+P in TUI)
static PROFILING_STATE: AtomicBool = AtomicBool::new(false);

/// Get the current profiling state (real-time, not cached)
#[inline]
pub fn get_profiling_state() -> bool {
    PROFILING_STATE.load(Ordering::Relaxed)
}

/// Set the profiling state
#[inline]
pub fn set_profiling_state(enabled: bool) {
    PROFILING_STATE.store(enabled, Ordering::Relaxed);
}

/// Toggle the profiling state and return the previous value
#[inline]
pub fn toggle_profiling_state() -> bool {
    PROFILING_STATE.fetch_xor(true, Ordering::Relaxed)
}

#[inline]
pub(crate) fn record_cb_gpu_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Keep this env var as an override.
        //
        // Default behavior must be conservative for performance:
        // - Attaching completion handlers for every command buffer can materially impact throughput
        //   in tight decode loops (especially when decode-batching is disabled / batch_size=1).
        // - Enable explicitly via env var or when profiling is enabled.
        RECORD_CB_GPU_TIMING.get().ok().flatten().unwrap_or_else(|| {
            use metallic_instrumentation::config::AppConfig;
            AppConfig::try_global().map(|c| c.enable_profiling).unwrap_or(false)
        })
    })
}

#[inline]
pub(crate) fn foundry_metrics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let cfg = match metallic_instrumentation::config::AppConfig::get_or_init_from_env() {
            Ok(cfg) => cfg,
            Err(_) => return false,
        };

        // Emit metrics whenever we have a sink configured, OR if we are in TUI mode (signaled via env).
        //
        // NOTE: This is intentionally *not* gated by `cfg.enable_profiling`. Context emits command-buffer timing
        // metrics in normal throughput mode (profiling off) when JSONL/console sinks are enabled, and we want
        // the same for Foundry so we can diagnose batching vs GPU kernel time deltas without forcing per-kernel
        // synchronization.
        cfg.enable_console_metrics || cfg.metrics_jsonl_path.is_some() || is_set(FoundryEnvVar::TuiMode)
    })
}

#[inline]
pub(crate) fn emit_cb_timing_metrics() -> bool {
    foundry_metrics_enabled() && record_cb_gpu_timing_enabled()
}

/// Controls whether the Foundry executor should disable batching and force per-kernel synchronization
/// while profiling is enabled.
///
/// Returns true if:
/// 1. Runtime profiling is enabled via `get_profiling_state()` (e.g., Ctrl+P in TUI), OR
/// 2. `METALLIC_FOUNDRY_PER_KERNEL_PROFILING=1` and profiling is enabled at startup
///
/// This is checked dynamically (not cached) to support runtime toggling.
#[inline]
pub(crate) fn foundry_per_kernel_profiling_enabled() -> bool {
    // Check Foundry's real-time profiling state first (toggled by Ctrl+P in TUI)
    if get_profiling_state() {
        return true;
    }

    // Fall back to env var + config check (cached internally by OnceLock)
    static ENV_OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    let env_override = *ENV_OVERRIDE.get_or_init(|| FOUNDRY_PER_KERNEL_PROFILING.get().ok().flatten());

    if let Some(explicit) = env_override {
        use metallic_instrumentation::config::AppConfig;
        let cfg = AppConfig::try_global();
        let profiling_enabled = cfg.map(|c| c.enable_profiling).unwrap_or(false);
        return explicit && profiling_enabled && foundry_metrics_enabled();
    }

    false
}

static CAPTURE_ID: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn next_capture_id() -> u64 {
    CAPTURE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub(crate) struct CaptureMetrics {
    pub(crate) id: u64,
    pub(crate) kernel_counts: FxHashMap<String, u32>,
    pub(crate) dispatches: u32,
}

impl CaptureMetrics {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            kernel_counts: FxHashMap::default(),
            dispatches: 0,
        }
    }

    pub(crate) fn record_kernel(&mut self, name: &str) {
        self.dispatches = self.dispatches.saturating_add(1);
        if let Some(slot) = self.kernel_counts.get_mut(name) {
            *slot = slot.saturating_add(1);
            return;
        }
        self.kernel_counts.insert(name.to_string(), 1);
    }
}

pub(crate) fn summarize_kernel_counts(metrics: &CaptureMetrics, max_entries: usize) -> FxHashMap<String, String> {
    let mut summary: FxHashMap<String, String> = FxHashMap::default();
    summary.insert("dispatches".to_string(), metrics.dispatches.to_string());

    if metrics.kernel_counts.is_empty() || max_entries == 0 {
        return summary;
    }

    let mut pairs: Vec<(&String, u32)> = metrics.kernel_counts.iter().map(|(k, v)| (k, *v)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    for (idx, (name, count)) in pairs.into_iter().take(max_entries).enumerate() {
        summary.insert(format!("k{:02}", idx), format!("{name}:{count}"));
    }

    summary
}

#[path = "instrument.test.rs"]
mod tests;
