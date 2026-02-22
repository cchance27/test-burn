#![cfg(test)]

use super::*;

#[test]
fn q6_k_spec_matches_ggml_block_constants() {
    assert_eq!(Q6_K_SPEC.weights_per_block, 256);
    assert_eq!(Q6_K_SPEC.block_bytes, 210);
}

#[test]
fn q6_k_storage_size_matches_known_tensor_shape() {
    let size = quantized_tensor_storage_bytes_for_dtype(Dtype::Q6_K, &[4864, 896]).expect("q6_k size");
    assert_eq!(size, 3_575_040);
}
