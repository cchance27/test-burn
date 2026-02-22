use std::sync::OnceLock;

pub const OP_METRICS_ENV_KEY: &str = "METALLIC_DEBUG_OP_METRICS";

/// Shared boolean env parser for debug feature toggles across crates.
#[must_use]
pub fn parse_env_bool(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
}

/// One-time lookup for op-metrics instrumentation toggle.
pub fn op_metrics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(OP_METRICS_ENV_KEY).ok().is_some_and(|v| parse_env_bool(&v)))
}
