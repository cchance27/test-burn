#![cfg(test)]

use super::*;

#[test]
fn test_embedding_params_metal_struct() {
    let def = EmbeddingParams::METAL_STRUCT_DEF;
    assert!(def.contains("struct EmbeddingParams"));
    assert!(def.contains("d_model"));
    assert!(def.contains("total_elements"));
    assert!(def.contains("vocab_size"));
}
