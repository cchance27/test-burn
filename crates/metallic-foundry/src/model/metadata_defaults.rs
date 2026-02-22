use metallic_loader::{LoadedModel, ModelMetadata};

use crate::{
    error::MetalError, spec::{ArchValue, ArchitectureDefaults, MetadataKeysSpec}
};

fn metadata_usize(metadata: &dyn ModelMetadata, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(val) = metadata.get_u32(key) {
            return Some(val as usize);
        }
    }
    None
}

fn metadata_f32(metadata: &dyn ModelMetadata, keys: &[&str]) -> Option<f32> {
    for key in keys {
        if let Some(val) = metadata.get_f32(key) {
            return Some(val);
        }
    }
    None
}

/// Infers architecture parameters from GGUF metadata based on a provided key mapping.
///
/// This function is entirely driven by the `MetadataKeysSpec`, which maps
/// generic architecture field names (e.g. "d_model") to an ordered list of
/// GGUF metadata keys to search.
pub fn infer_from_metadata_with_keys(
    metadata: &dyn ModelMetadata,
    keys_spec: &MetadataKeysSpec,
) -> Result<ArchitectureDefaults, MetalError> {
    let mut values = rustc_hash::FxHashMap::default();

    for (field, keys) in &keys_spec.keys {
        let keys_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

        // Check for special "array length" key pattern (e.g. "@len:tokenizer.ggml.tokens")
        for key in &keys_strs {
            if let Some(path) = key.strip_prefix("@len:")
                && let Some(arr) = metadata.get_array(path)
            {
                values.insert(field.clone(), ArchValue::USize(arr.len()));
                break;
            }
        }

        if values.contains_key(field) {
            continue;
        }

        // Standard value lookup
        if let Some(val) = metadata_usize(metadata, &keys_strs) {
            values.insert(field.clone(), ArchValue::USize(val));
        } else if let Some(val) = metadata_f32(metadata, &keys_strs) {
            values.insert(field.clone(), ArchValue::F32(val));
        }
    }

    Ok(ArchitectureDefaults { values })
}

/// Infers baseline architecture defaults from model using its internal knowledge.
///
/// This is used as a fallback when a model spec does not provide `architecture.metadata_keys`.
pub fn infer_architecture_defaults(model: &dyn LoadedModel) -> Result<ArchitectureDefaults, MetalError> {
    let mut values = rustc_hash::FxHashMap::default();

    for (field, val) in model.inferred_architecture_params() {
        match val {
            metallic_loader::MetadataValue::Int(i) => {
                values.insert(field, ArchValue::USize(i as usize));
            }
            metallic_loader::MetadataValue::Float(f) => {
                values.insert(field, ArchValue::F32(f as f32));
            }
            _ => {
                // Ignore non-numeric defaults for now
            }
        }
    }

    Ok(ArchitectureDefaults { values })
}

#[path = "metadata_defaults.test.rs"]
mod tests;
