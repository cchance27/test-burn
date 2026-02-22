#![cfg(test)]

use metallic_loader::{DummyMetadata, MapMetadata, MetadataValue};
use rustc_hash::FxHashMap;

use super::*;

struct MockModel {
    arch: String,
    params: Vec<(String, MetadataValue<'static>)>,
}

impl LoadedModel for MockModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn architecture(&self) -> Option<&str> {
        Some(&self.arch)
    }
    fn metadata(&self) -> &dyn ModelMetadata {
        &DummyMetadata
    }
    fn tensor_info(&self, _name: &str) -> Option<metallic_loader::TensorInfo> {
        None
    }
    fn tensor_data(&self, _name: &str) -> Result<metallic_loader::TensorData<'_>, metallic_loader::LoaderError> {
        Err(metallic_loader::LoaderError::TensorNotFound(_name.to_string()))
    }
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn estimated_memory_usage(&self) -> usize {
        0
    }
    fn offload_tensor(&self, _name: &str) -> Result<(), metallic_loader::LoaderError> {
        Ok(())
    }
    fn load_tensor(&self, _name: &str) -> Result<(), metallic_loader::LoaderError> {
        Ok(())
    }
    fn available_fallbacks(&self) -> &[String] {
        &[]
    }
    fn get_fallback(&self, _key: &str) -> Result<Option<metallic_loader::TensorData<'_>>, metallic_loader::LoaderError> {
        Ok(None)
    }
    fn inferred_architecture_params(&self) -> Vec<(String, MetadataValue<'_>)> {
        self.params.clone()
    }
}

#[test]
fn test_generic_inference() {
    let model = MockModel {
        arch: "test".to_string(),
        params: vec![
            ("d_model".to_string(), MetadataValue::Int(512)),
            ("vocab_size".to_string(), MetadataValue::Int(32000)),
        ],
    };

    let defaults = infer_architecture_defaults(&model).unwrap();

    assert_eq!(defaults.get_usize("d_model"), Some(512));
    assert_eq!(defaults.get_usize("vocab_size"), Some(32000));
}

#[test]
fn test_infer_from_metadata_with_llama_keys() {
    let mut entries = FxHashMap::default();
    entries.insert("llama.embedding_length".to_string(), MetadataValue::Int(3072));
    entries.insert("llama.attention.head_count".to_string(), MetadataValue::Int(24));
    entries.insert("llama.attention.head_count_kv".to_string(), MetadataValue::Int(8));
    entries.insert("llama.block_count".to_string(), MetadataValue::Int(28));
    entries.insert("llama.feed_forward_length".to_string(), MetadataValue::Int(8192));
    entries.insert("llama.context_length".to_string(), MetadataValue::Int(131072));
    entries.insert("llama.vocab_size".to_string(), MetadataValue::Int(128256));
    entries.insert("llama.rope.freq_base".to_string(), MetadataValue::Float(500000.0));
    entries.insert("llama.attention.layer_norm_rms_epsilon".to_string(), MetadataValue::Float(1e-5));
    let metadata = MapMetadata { entries };

    let mut keys = FxHashMap::default();
    keys.insert("d_model".to_string(), vec!["llama.embedding_length".to_string()]);
    keys.insert("n_heads".to_string(), vec!["llama.attention.head_count".to_string()]);
    keys.insert("n_kv_heads".to_string(), vec!["llama.attention.head_count_kv".to_string()]);
    keys.insert("n_layers".to_string(), vec!["llama.block_count".to_string()]);
    keys.insert("ff_dim".to_string(), vec!["llama.feed_forward_length".to_string()]);
    keys.insert("max_seq_len".to_string(), vec!["llama.context_length".to_string()]);
    keys.insert("vocab_size".to_string(), vec!["llama.vocab_size".to_string()]);
    keys.insert("rope_base".to_string(), vec!["llama.rope.freq_base".to_string()]);
    keys.insert("rms_eps".to_string(), vec!["llama.attention.layer_norm_rms_epsilon".to_string()]);
    let keys_spec = MetadataKeysSpec { keys };

    let defaults = infer_from_metadata_with_keys(&metadata, &keys_spec).unwrap();
    assert_eq!(defaults.get_usize("d_model"), Some(3072));
    assert_eq!(defaults.get_usize("n_heads"), Some(24));
    assert_eq!(defaults.get_usize("n_kv_heads"), Some(8));
    assert_eq!(defaults.get_usize("n_layers"), Some(28));
    assert_eq!(defaults.get_usize("ff_dim"), Some(8192));
    assert_eq!(defaults.get_usize("max_seq_len"), Some(131072));
    assert_eq!(defaults.get_usize("vocab_size"), Some(128256));
    assert_eq!(defaults.get_f32("rope_base"), Some(500000.0));
    assert_eq!(defaults.get_f32("rms_eps"), Some(1e-5));
}
