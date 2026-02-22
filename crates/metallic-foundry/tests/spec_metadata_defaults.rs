use metallic_foundry::{model, spec::ModelSpec};
use metallic_loader::{MapMetadata, MetadataValue, MockModel};
use rustc_hash::FxHashMap;

fn qwen2_min_metadata() -> MapMetadata {
    let mut entries = FxHashMap::default();
    entries.insert("general.architecture".into(), MetadataValue::String("qwen2".into()));
    entries.insert("qwen2.embedding_length".into(), MetadataValue::Int(896));
    entries.insert("qwen2.attention.head_count".into(), MetadataValue::Int(14));
    entries.insert("qwen2.attention.head_count_kv".into(), MetadataValue::Int(2));
    entries.insert("qwen2.block_count".into(), MetadataValue::Int(24));
    entries.insert("qwen2.feed_forward_length".into(), MetadataValue::Int(4864));
    entries.insert("qwen2.context_length".into(), MetadataValue::Int(32768));
    entries.insert("model.vocab_size".into(), MetadataValue::Int(151936));
    entries.insert("qwen2.rope.freq_base".into(), MetadataValue::Float(1e6));
    entries.insert("qwen2.attention.layer_norm_rms_epsilon".into(), MetadataValue::Float(1e-6));
    MapMetadata { entries }
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
    let metadata = qwen2_min_metadata();
    let mock = MockModel {
        architecture: Some("qwen2".to_string()),
        metadata,
        inferred_params: vec![
            ("d_model".to_string(), MetadataValue::Int(896)),
            ("n_heads".to_string(), MetadataValue::Int(14)),
            ("n_kv_heads".to_string(), MetadataValue::Int(2)),
            ("n_layers".to_string(), MetadataValue::Int(24)),
            ("ff_dim".to_string(), MetadataValue::Int(4864)),
            ("vocab_size".to_string(), MetadataValue::Int(151936)),
            ("max_seq_len".to_string(), MetadataValue::Int(32768)),
            ("rope_base".to_string(), MetadataValue::Float(1e6)),
        ],
    };
    let defaults = model::infer_architecture_defaults(&mock).unwrap();
    spec.architecture.apply_metadata_baseline(&defaults).unwrap();
    assert_eq!(spec.architecture.d_model(), 896);
    assert_eq!(spec.architecture.n_heads(), 14);
    assert_eq!(spec.architecture.n_kv_heads(), 2);
    assert_eq!(spec.architecture.n_layers(), 24);
    assert_eq!(spec.architecture.ff_dim(), 4864);
    assert_eq!(spec.architecture.vocab_size(), 151936);
    assert_eq!(spec.architecture.max_seq_len(), 32768);
    assert_eq!(spec.architecture.rope_base(), 1e6);
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
    let metadata = qwen2_min_metadata();
    let mock = MockModel {
        architecture: Some("qwen2".to_string()),
        metadata,
        inferred_params: vec![("d_model".to_string(), MetadataValue::Int(896))],
    };
    let defaults = model::infer_architecture_defaults(&mock).unwrap();
    spec.architecture.apply_metadata_baseline(&defaults).unwrap();
    assert_eq!(spec.architecture.d_model(), 112);
}
