#![cfg(test)]

use super::*;

#[test]
fn test_sample_params_metal_struct() {
    let def = SampleParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct SampleParams"));
    assert!(def.contains("vocab_size"));
    assert!(def.contains("per_thread_m"));
}

#[test]
fn top_k_zero_is_treated_as_greedy() {
    assert_eq!(SampleTopK::normalize_top_k(0, 1024), 1);
}

#[test]
fn top_k_is_clamped_to_vocab_size() {
    assert_eq!(SampleTopK::normalize_top_k(4096, 1024), 1024);
}
