use metallic_foundry::{
    gguf::file::{GGUFMetadata, GGUFValue}, model, spec::ModelSpec
};
use rustc_hash::FxHashMap;

fn qwen2_min_metadata() -> GGUFMetadata {
    let mut entries = FxHashMap::default();
    entries.insert("general.architecture".into(), GGUFValue::String("qwen2".into()));
    entries.insert("qwen2.embedding_length".into(), GGUFValue::U32(896));
    entries.insert("qwen2.attention.head_count".into(), GGUFValue::U32(14));
    entries.insert("qwen2.attention.head_count_kv".into(), GGUFValue::U32(2));
    entries.insert("qwen2.block_count".into(), GGUFValue::U32(24));
    entries.insert("qwen2.feed_forward_length".into(), GGUFValue::U32(4864));
    entries.insert("qwen2.context_length".into(), GGUFValue::U32(32768));
    entries.insert("model.vocab_size".into(), GGUFValue::U32(151936));
    entries.insert("qwen2.rope.freq_base".into(), GGUFValue::F32(1e6));
    entries.insert("qwen2.attention.layer_norm_rms_epsilon".into(), GGUFValue::F32(1e-6));
    GGUFMetadata { entries }
}

#[test]
fn spec_can_omit_arch_numerics_and_fill_from_metadata() {
    let json = r#"{
        "name": "test",
        "architecture": {
            "tensor_names": {},
            "prepare": {},
            "forward": []
        }
    }"#;
    let mut spec = ModelSpec::from_json(json).unwrap();
    assert_eq!(spec.architecture.d_model, 0);
    let defaults = model::infer_architecture_defaults_from_gguf_metadata(&qwen2_min_metadata()).unwrap();
    spec.architecture.apply_metadata_baseline(&defaults).unwrap();
    assert_eq!(spec.architecture.d_model, 896);
    assert_eq!(spec.architecture.n_heads, 14);
    assert_eq!(spec.architecture.n_kv_heads, 2);
    assert_eq!(spec.architecture.n_layers, 24);
    assert_eq!(spec.architecture.ff_dim, 4864);
    assert_eq!(spec.architecture.vocab_size, 151936);
    assert_eq!(spec.architecture.max_seq_len, 32768);
    assert_eq!(spec.architecture.rope_base, 1e6);
}

#[test]
fn dsl_override_wins_over_metadata_baseline() {
    let json = r#"{
        "name": "test",
        "architecture": {
            "d_model": 112,
            "tensor_names": {},
            "prepare": {},
            "forward": []
        }
    }"#;
    let mut spec = ModelSpec::from_json(json).unwrap();
    let defaults = model::infer_architecture_defaults_from_gguf_metadata(&qwen2_min_metadata()).unwrap();
    spec.architecture.apply_metadata_baseline(&defaults).unwrap();
    assert_eq!(spec.architecture.d_model, 112);
}
