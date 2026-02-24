#![cfg(test)]

use std::str::FromStr;

use super::*;

#[test]
fn dense_variant_parsing_accepts_expected_aliases() {
    assert_eq!(
        DenseStorageVariant::from_str("f16").expect("f16 should parse"),
        DenseStorageVariant::F16
    );
    assert_eq!(
        DenseStorageVariant::from_str("preserve_dense").expect("preserve_dense should parse"),
        DenseStorageVariant::Preserve
    );
}

#[test]
fn dense_variant_parsing_rejects_legacy_aliases() {
    let err = DenseStorageVariant::from_str("legacy").expect_err("legacy should be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unsupported dense storage variant"),
        "error should mention unsupported variant"
    );
}

#[test]
fn policy_variant_parsing_accepts_compact_and_kv_forms() {
    let compact = PolicyVariant::from_str("preserve:f32").expect("compact preserve form should parse");
    assert_eq!(compact.dense_storage, DenseStorageVariant::Preserve);
    assert_eq!(compact.quant_compute, QuantComputeVariant::F32);

    let dense_colon = PolicyVariant::from_str("dense:f16").expect("compact dense form should parse");
    assert_eq!(dense_colon.dense_storage, DenseStorageVariant::F16);
    assert_eq!(dense_colon.quant_compute, QuantComputeVariant::F16);

    let kv = PolicyVariant::from_str("dense=preserve,quant=bf16").expect("kv policy variant should parse");
    assert_eq!(kv.dense_storage, DenseStorageVariant::Preserve);
    assert_eq!(kv.quant_compute, QuantComputeVariant::BF16);
}

#[test]
fn policy_variant_parsing_rejects_unknown_keys() {
    let err = PolicyVariant::from_str("foo=bar").expect_err("unknown key should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unsupported policy variant key"),
        "error should mention unsupported key"
    );
}

#[test]
fn resolve_f32_f16_caps_storage_to_f16() {
    let resolved = resolve_policy_for_dtype(
        Dtype::F32,
        PolicyVariant {
            dense_storage: DenseStorageVariant::F16,
            quant_compute: QuantComputeVariant::F16,
        },
    )
    .expect("f32 cap f16 should resolve");
    assert_eq!(resolved.policy.short_name(), "f32");
    assert_eq!(resolved.resolution.storage_dtype, Dtype::F16);
    assert_eq!(resolved.resolution.compute_dtype, Dtype::F16);
    assert!(resolved.resolution.lossy_cast);
}

#[test]
fn resolve_f32_preserve_uses_native_policy() {
    let resolved =
        resolve_policy_for_dtype(Dtype::F32, PolicyVariant::preserve_dense(QuantComputeVariant::F16)).expect("f32 preserve should resolve");
    assert_eq!(resolved.policy.short_name(), "f32_native");
    assert_eq!(resolved.resolution.storage_dtype, Dtype::F32);
    assert_eq!(resolved.resolution.compute_dtype, Dtype::F32);
    assert!(!resolved.resolution.lossy_cast);
}

#[test]
fn resolve_q8_allows_compute_variant_metadata() {
    let resolved = resolve_policy_for_dtype(
        Dtype::Q8_0,
        PolicyVariant {
            dense_storage: DenseStorageVariant::Preserve,
            quant_compute: QuantComputeVariant::BF16,
        },
    )
    .expect("q8 should resolve");
    assert_eq!(resolved.policy.short_name(), "q8");
    assert_eq!(resolved.resolution.storage_dtype, Dtype::Q8_0);
    assert_eq!(resolved.resolution.compute_dtype, Dtype::BF16);
}

#[test]
fn unsupported_dtype_is_fail_fast() {
    let err = resolve_policy_for_dtype(Dtype::BF16, PolicyVariant::preserve_dense(QuantComputeVariant::F16))
        .expect_err("bf16 should be unsupported for now");
    let msg = format!("{err:#}");
    assert!(msg.contains("Unsupported tensor dtype"), "error should mention unsupported dtype");
}
