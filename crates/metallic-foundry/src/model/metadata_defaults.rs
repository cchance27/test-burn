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
