use std::sync::{
    OnceLock, atomic::{AtomicU64, Ordering}
};

use rustc_hash::FxHashMap;

pub(crate) const METALLIC_RECORD_CB_GPU_TIMING_ENV: &str = "METALLIC_RECORD_CB_GPU_TIMING";
pub(crate) const METALLIC_FOUNDRY_PER_KERNEL_PROFILING_ENV: &str = "METALLIC_FOUNDRY_PER_KERNEL_PROFILING";

#[inline]
fn parse_env_bool(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
}

#[inline]
pub(crate) fn record_cb_gpu_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Keep this env var as an override, but default to enabled when profiling+metrics are enabled.
        // This matches Context's behavior where `METALLIC_ENABLE_PROFILING=1` is sufficient to emit
        // per-command-buffer timing metrics (and disable batching when requested by the executor).
        std::env::var(METALLIC_RECORD_CB_GPU_TIMING_ENV)
            .ok()
            .map(|value| parse_env_bool(&value))
            .unwrap_or(true)
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

        // Emit metrics whenever we have a sink configured.
        //
        // NOTE: This is intentionally *not* gated by `cfg.enable_profiling`. Context emits command-buffer timing
        // metrics in normal throughput mode (profiling off) when JSONL/console sinks are enabled, and we want
        // the same for Foundry so we can diagnose batching vs GPU kernel time deltas without forcing per-kernel
        // synchronization.
        cfg.enable_console_metrics || cfg.metrics_jsonl_path.is_some()
    })
}

#[inline]
pub(crate) fn emit_cb_timing_metrics() -> bool {
    foundry_metrics_enabled() && record_cb_gpu_timing_enabled()
}

/// Controls whether the Foundry executor should disable batching and force per-kernel synchronization
/// while profiling is enabled.
///
/// Defaults to enabled to match Context's profiling behavior. Set
/// `METALLIC_FOUNDRY_PER_KERNEL_PROFILING=0` to keep batching enabled while still emitting metrics.
#[inline]
pub(crate) fn foundry_per_kernel_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let cfg = match metallic_instrumentation::config::AppConfig::get_or_init_from_env() {
            Ok(cfg) => cfg,
            Err(_) => return false,
        };
        if !cfg.enable_profiling || !foundry_metrics_enabled() {
            return false;
        }
        std::env::var(METALLIC_FOUNDRY_PER_KERNEL_PROFILING_ENV)
            .ok()
            .map(|value| parse_env_bool(&value))
            .unwrap_or(true)
    })
}

static CAPTURE_ID: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn next_capture_id() -> u64 {
    CAPTURE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub(crate) struct CaptureMetrics {
    pub(crate) id: u64,
    pub(crate) kernel_counts: FxHashMap<&'static str, u32>,
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

    pub(crate) fn record_kernel(&mut self, name: &'static str) {
        self.dispatches = self.dispatches.saturating_add(1);
        let slot = self.kernel_counts.entry(name).or_insert(0);
        *slot = slot.saturating_add(1);
    }
}

pub(crate) fn summarize_kernel_counts(metrics: &CaptureMetrics, max_entries: usize) -> FxHashMap<String, String> {
    let mut summary: FxHashMap<String, String> = FxHashMap::default();
    summary.insert("dispatches".to_string(), metrics.dispatches.to_string());

    if metrics.kernel_counts.is_empty() || max_entries == 0 {
        return summary;
    }

    let mut pairs: Vec<(&'static str, u32)> = metrics.kernel_counts.iter().map(|(k, v)| (*k, *v)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    for (idx, (name, count)) in pairs.into_iter().take(max_entries).enumerate() {
        summary.insert(format!("k{:02}", idx), format!("{name}:{count}"));
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_bool() {
        for value in ["1", "true", "TRUE", "yes", "on", " 1 ", " tRuE "] {
            assert!(parse_env_bool(value), "expected truthy: {value}");
        }

        for value in ["", " ", "0", "false", "FALSE", "no", "off", " 0 "] {
            assert!(!parse_env_bool(value), "expected falsy: {value}");
        }
    }

    #[test]
    fn test_summarize_kernel_counts_is_deterministic() {
        let mut metrics = CaptureMetrics::new(0);
        metrics.record_kernel("b");
        metrics.record_kernel("a");
        metrics.record_kernel("b");

        let summary = summarize_kernel_counts(&metrics, 8);
        assert_eq!(summary.get("dispatches").unwrap(), "3");
        assert_eq!(summary.get("k00").unwrap(), "b:2");
        assert_eq!(summary.get("k01").unwrap(), "a:1");
    }
}
