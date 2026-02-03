use crate::{
    error::MetalError, gguf::file::{GGUFMetadata, GGUFValue}, spec::{ArchitectureDefaults, MetadataKeysSpec, MetadataValue}
};

fn metadata_usize(metadata: &GGUFMetadata, keys: &[&str]) -> Option<usize> {
    for key in keys {
        let Some(value) = metadata.entries.get(*key) else { continue };
        match value {
            GGUFValue::U32(v) => return Some(*v as usize),
            GGUFValue::U64(v) => return Some(*v as usize),
            GGUFValue::I32(v) if *v >= 0 => return Some(*v as usize),
            GGUFValue::I64(v) if *v >= 0 => return Some(*v as usize),
            _ => {}
        }
    }
    None
}

fn metadata_f32(metadata: &GGUFMetadata, keys: &[&str]) -> Option<f32> {
    for key in keys {
        let Some(value) = metadata.entries.get(*key) else { continue };
        match value {
            GGUFValue::F32(v) => return Some(*v),
            GGUFValue::F64(v) => return Some(*v as f32),
            _ => {}
        }
    }
    None
}

/// Infers architecture parameters from GGUF metadata based on a provided key mapping.
///
/// This function is entirely driven by the `MetadataKeysSpec`, which maps
/// generic architecture field names (e.g. "d_model") to an ordered list of
/// GGUF metadata keys to search.
pub fn infer_from_gguf_with_keys(metadata: &GGUFMetadata, keys_spec: &MetadataKeysSpec) -> Result<ArchitectureDefaults, MetalError> {
    let mut values = rustc_hash::FxHashMap::default();

    for (field, keys) in &keys_spec.keys {
        let keys_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

        // Check for special "array length" key pattern (e.g. "@len:tokenizer.ggml.tokens")
        for key in &keys_strs {
            if let Some(path) = key.strip_prefix("@len:") {
                if let Some(GGUFValue::Array(arr)) = metadata.entries.get(path) {
                    values.insert(field.clone(), MetadataValue::USize(arr.len()));
                    break;
                }
            }
        }

        if values.contains_key(field) {
            continue;
        }

        // Standard value lookup
        if let Some(val) = metadata_usize(metadata, &keys_strs) {
            values.insert(field.clone(), MetadataValue::USize(val));
        } else if let Some(val) = metadata_f32(metadata, &keys_strs) {
            values.insert(field.clone(), MetadataValue::F32(val));
        }
    }

    Ok(ArchitectureDefaults { values })
}

/// Infers baseline architecture defaults from GGUF metadata using built-in key mappings.
///
/// This is used as a fallback when a model spec does not provide `architecture.metadata_keys`.
/// Keep this small and focused: prefer explicit key mappings in the DSL for non-standard models.
pub fn infer_architecture_defaults_from_gguf_metadata(metadata: &GGUFMetadata) -> Result<ArchitectureDefaults, MetalError> {
    let arch = metadata
        .entries
        .get("general.architecture")
        .and_then(|v| match v {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");

    let mut keys: rustc_hash::FxHashMap<String, Vec<String>> = rustc_hash::FxHashMap::default();

    // Common keys (many GGUFs expose these).
    keys.insert("vocab_size".into(), vec!["model.vocab_size".into()]);

    if arch.contains("qwen2") {
        keys.insert("d_model".into(), vec!["qwen2.embedding_length".into()]);
        keys.insert("n_heads".into(), vec!["qwen2.attention.head_count".into()]);
        keys.insert("n_kv_heads".into(), vec!["qwen2.attention.head_count_kv".into()]);
        keys.insert("n_layers".into(), vec!["qwen2.block_count".into()]);
        keys.insert("ff_dim".into(), vec!["qwen2.feed_forward_length".into()]);
        keys.insert("max_seq_len".into(), vec!["qwen2.context_length".into()]);
        keys.insert("rope_base".into(), vec!["qwen2.rope.freq_base".into()]);
        keys.insert("rms_eps".into(), vec!["qwen2.attention.layer_norm_rms_epsilon".into()]);
    } else if !arch.is_empty() {
        return Err(MetalError::InvalidOperation(format!(
            "No built-in metadata key mapping for architecture '{arch}'. Provide `architecture.metadata_keys` in the model spec."
        )));
    }

    infer_from_gguf_with_keys(metadata, &MetadataKeysSpec { keys })
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;

    #[test]
    fn test_generic_inference() {
        let mut metadata = GGUFMetadata {
            entries: FxHashMap::default(),
        };
        metadata.entries.insert("test.dim".into(), GGUFValue::U32(512));
        metadata
            .entries
            .insert("test.array".into(), GGUFValue::Array(vec![GGUFValue::U32(1), GGUFValue::U32(2)]));

        let mut keys = FxHashMap::default();
        keys.insert("d_model".into(), vec!["test.dim".into()]);
        keys.insert("vocab_size".into(), vec!["@len:test.array".into()]);

        let spec = MetadataKeysSpec { keys };
        let defaults = infer_from_gguf_with_keys(&metadata, &spec).unwrap();

        assert_eq!(defaults.get_usize("d_model"), Some(512));
        assert_eq!(defaults.get_usize("vocab_size"), Some(2));
    }
}
