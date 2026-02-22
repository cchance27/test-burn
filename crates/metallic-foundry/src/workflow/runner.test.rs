#![cfg(test)]

use super::should_override_with_inferred_eos;

#[test]
fn inferred_eos_used_when_not_explicitly_supplied() {
    assert!(should_override_with_inferred_eos(false, Some(42), Some(42)));
    assert!(should_override_with_inferred_eos(false, None, Some(42)));
}

#[test]
fn inferred_eos_used_when_supplied_value_matches_workflow_default() {
    assert!(should_override_with_inferred_eos(true, Some(42), Some(42)));
}

#[test]
fn explicit_non_default_eos_is_preserved() {
    assert!(!should_override_with_inferred_eos(true, Some(2), Some(42)));
}
