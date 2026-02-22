#![cfg(test)]

use super::{GGUFMetadata, GGUFValue, adjust_embedding_dims};

#[test]
fn adjust_embedding_dims_uses_arch_prefixed_metadata() {
    let mut metadata = GGUFMetadata::default();
    metadata
        .entries
        .insert("general.architecture".to_string(), GGUFValue::String("customarch".to_string()));
    metadata
        .entries
        .insert("customarch.embedding_length".to_string(), GGUFValue::U32(1024));
    metadata.entries.insert("customarch.vocab_size".to_string(), GGUFValue::U32(32000));

    let mut dims = [1024usize, 32000usize];
    adjust_embedding_dims("token_embd.weight", &mut dims, &metadata, Some("customarch"));
    assert_eq!(dims, [32000, 1024]);
}

#[test]
fn adjust_embedding_dims_falls_back_to_model_keys() {
    let mut metadata = GGUFMetadata::default();
    metadata.entries.insert("model.d_model".to_string(), GGUFValue::U32(2048));
    metadata.entries.insert("model.vocab_size".to_string(), GGUFValue::U32(128000));

    let mut dims = [2048usize, 128000usize];
    adjust_embedding_dims("token_embd.weight", &mut dims, &metadata, None);
    assert_eq!(dims, [128000, 2048]);
}

#[test]
fn adjust_embedding_dims_leaves_non_matching_shapes_unchanged() {
    let mut metadata = GGUFMetadata::default();
    metadata.entries.insert("model.d_model".to_string(), GGUFValue::U32(2048));
    metadata.entries.insert("model.vocab_size".to_string(), GGUFValue::U32(128000));

    let mut dims = [2048usize, 777usize];
    adjust_embedding_dims("layer.weight", &mut dims, &metadata, None);
    assert_eq!(dims, [2048, 777]);
}
