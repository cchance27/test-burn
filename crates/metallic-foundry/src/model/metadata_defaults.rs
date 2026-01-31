use crate::{
    error::MetalError, gguf::file::{GGUFMetadata, GGUFValue}, spec::{ArchitectureDefaults, MetadataKeysSpec}
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

fn metadata_vocab_size(metadata: &GGUFMetadata) -> Option<usize> {
    if let Some(v) = metadata_usize(metadata, &["vocab_size", "model.vocab_size"]) {
        return Some(v);
    }
    if let Some(GGUFValue::Array(values)) = metadata.entries.get("tokenizer.ggml.tokens") {
        return Some(values.len());
    }
    None
}

fn architecture_str(metadata: &GGUFMetadata) -> Option<&str> {
    metadata.entries.get("general.architecture").and_then(|v| match v {
        GGUFValue::String(s) => Some(s.as_str()),
        _ => None,
    })
}

fn require_usize(metadata: &GGUFMetadata, name: &str, keys: &[&str]) -> Result<usize, MetalError> {
    metadata_usize(metadata, keys).ok_or_else(|| {
        let arch = architecture_str(metadata).unwrap_or("<unknown>");
        MetalError::InvalidShape(format!(
            "GGUF metadata missing required '{name}' (arch={arch}). Looked for keys: {keys:?}"
        ))
    })
}

fn keys_from_spec<'a>(spec: Option<&'a MetadataKeysSpec>, field: &str, fallback: &[&'a str]) -> Vec<&'a str> {
    if let Some(spec) = spec
        && let Some(keys) = spec.keys.get(field)
        && !keys.is_empty()
    {
        return keys.iter().map(|s| s.as_str()).collect();
    }
    fallback.to_vec()
}

pub fn infer_from_gguf_with_keys(
    metadata: &GGUFMetadata,
    keys_spec: Option<&MetadataKeysSpec>,
) -> Result<ArchitectureDefaults, MetalError> {
    // If a model supplies `architecture.metadata_keys`, prefer it (first-match-wins per field).
    // Otherwise, fall back to a built-in key set for compatibility.
    let d_model = {
        let keys = keys_from_spec(
            keys_spec,
            "d_model",
            &[
                "qwen2.embedding_length",
                "qwen2.d_model",
                "llama.embedding_length",
                "llama.d_model",
                "model.d_model",
            ],
        );
        require_usize(metadata, "d_model", &keys)
    }?;
    let n_heads = {
        let keys = keys_from_spec(keys_spec, "n_heads", &["qwen2.attention.head_count", "llama.attention.head_count"]);
        require_usize(metadata, "n_heads", &keys)
    }?;
    let n_kv_heads = {
        let keys = keys_from_spec(
            keys_spec,
            "n_kv_heads",
            &["qwen2.attention.head_count_kv", "llama.attention.head_count_kv"],
        );
        require_usize(metadata, "n_kv_heads", &keys)
    }?;
    let n_layers = {
        let keys = keys_from_spec(keys_spec, "n_layers", &["qwen2.block_count", "llama.block_count"]);
        require_usize(metadata, "n_layers", &keys)
    }?;
    let ff_dim = {
        let keys = keys_from_spec(keys_spec, "ff_dim", &["qwen2.feed_forward_length", "llama.feed_forward_length"]);
        require_usize(metadata, "ff_dim", &keys)
    }?;
    let max_seq_len = {
        let keys = keys_from_spec(keys_spec, "max_seq_len", &["qwen2.context_length", "llama.context_length"]);
        require_usize(metadata, "max_seq_len", &keys)
    }?;

    let vocab_size = if let Some(keys_spec) = keys_spec
        && let Some(keys) = keys_spec.keys.get("vocab_size")
        && !keys.is_empty()
    {
        let keys: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        metadata_usize(metadata, &keys).or_else(|| {
            // Allow the tokenizer array fallback when explicitly requested via the well-known key.
            if keys.iter().any(|&k| k == "tokenizer.ggml.tokens") {
                metadata_vocab_size(metadata)
            } else {
                None
            }
        })
    } else {
        metadata_vocab_size(metadata)
    }
    .ok_or_else(|| {
        let arch = architecture_str(metadata).unwrap_or("<unknown>");
        MetalError::InvalidShape(format!(
            "GGUF metadata missing required 'vocab_size' (arch={arch}). Provide architecture.metadata_keys.keys.vocab_size or include tokenizer.ggml.tokens."
        ))
    })?;

    let rope_base = {
        let keys = keys_from_spec(keys_spec, "rope_base", &["qwen2.rope.freq_base", "llama.rope.freq_base"]);
        metadata_f32(metadata, &keys).unwrap_or(10000.0)
    };
    let rms_eps = {
        let keys = keys_from_spec(
            keys_spec,
            "rms_eps",
            &["qwen2.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"],
        );
        metadata_f32(metadata, &keys).unwrap_or(1e-6)
    };

    Ok(ArchitectureDefaults {
        d_model,
        n_heads,
        n_kv_heads,
        n_layers,
        ff_dim,
        vocab_size,
        max_seq_len,
        rope_base,
        rms_eps,
    })
}

pub fn infer_from_gguf(metadata: &GGUFMetadata) -> Result<ArchitectureDefaults, MetalError> {
    infer_from_gguf_with_keys(metadata, None)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::gguf::file::GGUFMetadata;

    #[test]
    fn infer_from_gguf_qwen2_keys() {
        let mut metadata = GGUFMetadata {
            entries: FxHashMap::default(),
        };
        metadata
            .entries
            .insert("general.architecture".into(), GGUFValue::String("qwen2".into()));
        metadata.entries.insert("qwen2.embedding_length".into(), GGUFValue::U32(896));
        metadata.entries.insert("qwen2.attention.head_count".into(), GGUFValue::U32(14));
        metadata.entries.insert("qwen2.attention.head_count_kv".into(), GGUFValue::U32(2));
        metadata.entries.insert("qwen2.block_count".into(), GGUFValue::U32(24));
        metadata.entries.insert("qwen2.feed_forward_length".into(), GGUFValue::U32(4864));
        metadata.entries.insert("qwen2.context_length".into(), GGUFValue::U32(32768));
        metadata.entries.insert("model.vocab_size".into(), GGUFValue::U32(151936));
        metadata.entries.insert("qwen2.rope.freq_base".into(), GGUFValue::F32(1e6));
        metadata
            .entries
            .insert("qwen2.attention.layer_norm_rms_epsilon".into(), GGUFValue::F32(1e-6));

        let defaults = infer_from_gguf(&metadata).unwrap();
        assert_eq!(defaults.d_model, 896);
        assert_eq!(defaults.n_heads, 14);
        assert_eq!(defaults.n_kv_heads, 2);
        assert_eq!(defaults.n_layers, 24);
        assert_eq!(defaults.ff_dim, 4864);
        assert_eq!(defaults.max_seq_len, 32768);
        assert_eq!(defaults.vocab_size, 151936);
        assert_eq!(defaults.rope_base, 1e6);
        assert_eq!(defaults.rms_eps, 1e-6);
    }

    #[test]
    fn infer_from_gguf_with_dsl_keys_prefers_spec() {
        let mut metadata = GGUFMetadata {
            entries: FxHashMap::default(),
        };
        metadata
            .entries
            .insert("general.architecture".into(), GGUFValue::String("custom".into()));
        metadata.entries.insert("custom.d_model".into(), GGUFValue::U32(123));
        metadata.entries.insert("custom.n_heads".into(), GGUFValue::U32(3));
        metadata.entries.insert("custom.n_kv_heads".into(), GGUFValue::U32(1));
        metadata.entries.insert("custom.n_layers".into(), GGUFValue::U32(2));
        metadata.entries.insert("custom.ff_dim".into(), GGUFValue::U32(4));
        metadata.entries.insert("custom.max_seq_len".into(), GGUFValue::U32(8));
        metadata.entries.insert("custom.vocab_size".into(), GGUFValue::U32(16));

        let mut keys = FxHashMap::default();
        keys.insert("d_model".into(), vec!["custom.d_model".into()]);
        keys.insert("n_heads".into(), vec!["custom.n_heads".into()]);
        keys.insert("n_kv_heads".into(), vec!["custom.n_kv_heads".into()]);
        keys.insert("n_layers".into(), vec!["custom.n_layers".into()]);
        keys.insert("ff_dim".into(), vec!["custom.ff_dim".into()]);
        keys.insert("max_seq_len".into(), vec!["custom.max_seq_len".into()]);
        keys.insert("vocab_size".into(), vec!["custom.vocab_size".into()]);
        let spec = MetadataKeysSpec { keys };

        let defaults = infer_from_gguf_with_keys(&metadata, Some(&spec)).unwrap();
        assert_eq!(defaults.d_model, 123);
        assert_eq!(defaults.n_heads, 3);
        assert_eq!(defaults.n_kv_heads, 1);
        assert_eq!(defaults.n_layers, 2);
        assert_eq!(defaults.ff_dim, 4);
        assert_eq!(defaults.max_seq_len, 8);
        assert_eq!(defaults.vocab_size, 16);
        // Optional fields fall back when missing.
        assert_eq!(defaults.rope_base, 10000.0);
        assert_eq!(defaults.rms_eps, 1e-6);
    }
}
