#![cfg(test)]

use metallic_sdk::parse_env_bool;

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
