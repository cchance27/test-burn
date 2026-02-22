#![cfg(test)]

use super::*;

#[test]
fn q6k_gguf_size_matches_known_shape() {
    let size = quantized_tensor_storage_bytes_for_gguf_dtype(GGUFDataType::Q6K, &[4864_u64, 896_u64]).expect("q6k bytes");
    assert_eq!(size, 3_575_040);
}

#[test]
fn iq4nl_block_spec_matches_reference_constants() {
    let spec = block_quant_spec_for_gguf_dtype(GGUFDataType::IQ4NL).expect("iq4nl spec");
    assert_eq!(spec.weights_per_block, 32);
    assert_eq!(spec.block_bytes, 18);
}

#[test]
fn deprecated_legacy_variants_are_classified_explicitly() {
    assert!(matches!(classify_gguf_dtype(GGUFDataType::Q4_2), GGUFDtypeClass::Deprecated { .. }));
    assert!(matches!(
        classify_gguf_dtype(GGUFDataType::Q4044),
        GGUFDtypeClass::Deprecated { .. }
    ));
    assert!(matches!(
        classify_gguf_dtype(GGUFDataType::IQ4NL44),
        GGUFDtypeClass::Deprecated { .. }
    ));
}
