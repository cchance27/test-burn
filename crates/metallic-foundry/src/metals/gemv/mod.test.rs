#![cfg(test)]

use super::{GemvStrategy, resolve_gemv_strategy_for_input};
use crate::{compound::Layout, tensor::Dtype};

#[test]
fn gemv_strategy_auto_uses_fast_path_for_f16() {
    let selected =
        resolve_gemv_strategy_for_input(GemvStrategy::Auto, Dtype::F16, Dtype::F16, Layout::RowMajor, 896).expect("strategy resolution");
    assert_eq!(selected, GemvStrategy::Auto);
}

#[test]
fn gemv_strategy_auto_downgrades_to_canonical_for_f32() {
    let selected =
        resolve_gemv_strategy_for_input(GemvStrategy::Auto, Dtype::F16, Dtype::F32, Layout::RowMajor, 896).expect("strategy resolution");
    assert_eq!(selected, GemvStrategy::Canonical);
}

#[test]
fn gemv_strategy_vectorized_rejects_non_f16() {
    let err = resolve_gemv_strategy_for_input(GemvStrategy::Vectorized, Dtype::F16, Dtype::F32, Layout::RowMajor, 896)
        .expect_err("expected fail-fast");
    let msg = format!("{err}");
    assert!(
        msg.contains("Vectorized/DecodeLmHead strategies require F16 input dtype"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gemv_strategy_auto_keeps_vectorized_for_colmajor_large_n() {
    let selected =
        resolve_gemv_strategy_for_input(GemvStrategy::Auto, Dtype::F16, Dtype::F16, Layout::ColMajor, 4096).expect("strategy resolution");
    assert_eq!(selected, GemvStrategy::Auto);
}

#[test]
fn gemv_strategy_scalar_rejects_rowmajor() {
    let err = resolve_gemv_strategy_for_input(GemvStrategy::Scalar, Dtype::F16, Dtype::F16, Layout::RowMajor, 896)
        .expect_err("expected fail-fast");
    let msg = format!("{err}");
    assert!(msg.contains("Scalar strategy only supports ColMajor"), "unexpected error: {msg}");
}

#[test]
fn gemv_strategy_auto_selects_decode_lmhead_for_large_rowmajor_f16() {
    let selected = resolve_gemv_strategy_for_input(GemvStrategy::Auto, Dtype::F16, Dtype::F16, Layout::RowMajor, 151_936)
        .expect("strategy resolution");
    assert_eq!(selected, GemvStrategy::DecodeLmHead);
}
