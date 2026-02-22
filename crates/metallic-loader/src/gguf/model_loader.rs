use std::{borrow::Cow, sync::Arc};

use rustc_hash::FxHashMap;

use super::{
    GGUFDataType, GGUFError, GGUFFile, file::{GGUFMetadata, GGUFValue}, tensor_info::GGUTensorInfo
};
use crate::{Dtype, LoadedModel, LoaderError, MetadataValue, ModelMetadata, TensorData, TensorInfo}; // Added Dtype

impl TryFrom<GGUFDataType> for Dtype {
    type Error = GGUFError;

    fn try_from(value: GGUFDataType) -> Result<Self, Self::Error> {
        match value {
            GGUFDataType::F32 => Ok(Dtype::F32),
            GGUFDataType::F16 => Ok(Dtype::F16),
            GGUFDataType::Q4_0 => Ok(Dtype::Q4_0),
            GGUFDataType::Q4_1 => Ok(Dtype::Q4_1),
            GGUFDataType::Q5_0 => Ok(Dtype::Q5_0),
            GGUFDataType::Q5_1 => Ok(Dtype::Q5_1),
            GGUFDataType::Q8_0 => Ok(Dtype::Q8_0),
            GGUFDataType::Q8_1 => Ok(Dtype::Q8_1),
            GGUFDataType::Q2K => Ok(Dtype::Q2_K), // Note: Mapping GGUF naming to SDK naming
            GGUFDataType::Q3K => Ok(Dtype::Q3_K),
            GGUFDataType::Q4K => Ok(Dtype::Q4_K),
            GGUFDataType::Q5K => Ok(Dtype::Q5_K),
            GGUFDataType::Q6K => Ok(Dtype::Q6_K),
            GGUFDataType::Q8K => Ok(Dtype::Q8_K),
            GGUFDataType::I8 => Ok(Dtype::I8),
            GGUFDataType::I16 => Ok(Dtype::I16),
            GGUFDataType::I32 => Ok(Dtype::I32),
            GGUFDataType::I64 => Err(GGUFError::UnsupportedDataType(format!("{value:?}"))), // No direct I64 in SDK Dtype yet? SDK defines F64 but not I64? SDK Dtype has F64.
            GGUFDataType::F64 => Ok(Dtype::F64),
            GGUFDataType::BF16 => Ok(Dtype::BF16),
            GGUFDataType::MXFP4 => Err(GGUFError::UnsupportedDataType(format!("{value:?}"))),
            _ => Err(GGUFError::UnsupportedDataType(format!("Unsupported GGUF type: {value:?}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGUFLayoutHint {
    Nk,
    Kn,
}

impl GGUFLayoutHint {
    fn from_architecture(arch: Option<&str>) -> Self {
        match arch.map(str::to_ascii_lowercase) {
            Some(ref a) if a.contains("falcon") || a.contains("refact") => GGUFLayoutHint::Kn,
            _ => GGUFLayoutHint::Nk,
        }
    }
}

const METALLIC_GGUF_LAYOUT_OVERRIDE_ENV: &str = "METALLIC_GGUF_LAYOUT_OVERRIDE";
const METALLIC_GGUF_LAYOUT_DEBUG_ENV: &str = "METALLIC_GGUF_LAYOUT_DEBUG";

fn parse_layout_hint(value: &str) -> Option<GGUFLayoutHint> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.contains("kn") {
        Some(GGUFLayoutHint::Kn)
    } else if normalized.contains("nk") {
        Some(GGUFLayoutHint::Nk)
    } else {
        None
    }
}

fn layout_hint_override() -> Option<GGUFLayoutHint> {
    let value = std::env::var(METALLIC_GGUF_LAYOUT_OVERRIDE_ENV).ok()?;
    parse_layout_hint(&value)
}

fn layout_hint_from_metadata_with_key(metadata: &GGUFMetadata, arch: Option<&str>) -> Option<(GGUFLayoutHint, String, String)> {
    let mut keys = Vec::new();
    keys.push("tensor_data_layout".to_string());
    keys.push("general.tensor_data_layout".to_string());
    if let Some(arch) = arch {
        let arch_lower = arch.to_ascii_lowercase();
        keys.push(format!("{arch_lower}.tensor_data_layout"));
    }

    for key in keys {
        if let Some(GGUFValue::String(value)) = metadata.entries.get(&key)
            && let Some(hint) = parse_layout_hint(value)
        {
            return Some((hint, key, value.clone()));
        }
    }

    for (key, value) in &metadata.entries {
        if key.ends_with(".tensor_data_layout")
            && let GGUFValue::String(layout) = value
            && let Some(hint) = parse_layout_hint(layout)
        {
            return Some((hint, key.clone(), layout.clone()));
        }
    }

    None
}

fn layout_debug_enabled() -> bool {
    std::env::var(METALLIC_GGUF_LAYOUT_DEBUG_ENV).ok().is_some_and(|v| v.trim() != "0")
}

fn log_layout_hint(arch: Option<&str>, hint: GGUFLayoutHint, source: Option<(String, String)>) {
    if !layout_debug_enabled() {
        return;
    }
    match source {
        Some((key, value)) => {
            eprintln!("[GGUF_LAYOUT] arch={arch:?} hint={hint:?} source_key={key} source_value={value}");
        }
        None => {
            eprintln!("[GGUF_LAYOUT] arch={arch:?} hint={hint:?} source_key=<none>");
        }
    }
}

fn metadata_usize(metadata: &GGUFMetadata, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(value) = metadata.entries.get(*key) {
            match value {
                GGUFValue::U32(v) => return Some(*v as usize),
                GGUFValue::U64(v) => return Some(*v as usize),
                GGUFValue::I32(v) if *v >= 0 => return Some(*v as usize),
                GGUFValue::I64(v) if *v >= 0 => return Some(*v as usize),
                _ => {}
            }
        }
    }
    None
}

fn metadata_architecture(metadata: &GGUFMetadata) -> Option<&str> {
    match metadata.entries.get("general.architecture") {
        Some(GGUFValue::String(value)) if !value.trim().is_empty() => Some(value.as_str()),
        _ => None,
    }
}

fn metadata_arch_usize(metadata: &GGUFMetadata, arch: Option<&str>, suffixes: &[&str], fallback_keys: &[&str]) -> Option<usize> {
    if let Some(arch) = arch {
        for suffix in suffixes {
            let key = format!("{arch}.{suffix}");
            if let Some(v) = metadata_usize(metadata, &[&key]) {
                return Some(v);
            }
        }
    }

    metadata_usize(metadata, fallback_keys)
}

fn metadata_vocab_size(metadata: &GGUFMetadata, arch: Option<&str>) -> Option<usize> {
    if let Some(v) = metadata_arch_usize(metadata, arch, &["vocab_size"], &["vocab_size", "model.vocab_size"]) {
        return Some(v);
    }

    if let Some(GGUFValue::Array(values)) = metadata.entries.get("tokenizer.ggml.tokens") {
        return Some(values.len());
    }

    None
}

fn adjust_embedding_dims(_name: &str, dims: &mut [usize], metadata: &GGUFMetadata, arch: Option<&str>) {
    if dims.len() != 2 {
        return;
    }

    let d_model = metadata_arch_usize(metadata, arch, &["embedding_length", "d_model"], &["model.d_model", "d_model"]);
    let vocab = metadata_vocab_size(metadata, arch);

    if let (Some(d_model), Some(vocab)) = (d_model, vocab)
        && dims[0] == d_model
        && dims[1] == vocab
    {
        dims.swap(0, 1);
    }
}

/// A model loader that can construct a Metallic model from GGUF tensors
pub struct GGUFModelLoader {
    pub(crate) gguf_file: Arc<GGUFFile>,
}

impl GGUFModelLoader {
    /// Create a new model loader from a GGUF file
    pub fn new(gguf_file: GGUFFile) -> Self {
        Self {
            gguf_file: Arc::new(gguf_file),
        }
    }

    /// Load a model from the GGUF file
    pub fn load_model(&self) -> Result<GGUFModel, GGUFError> {
        let mut tensors: FxHashMap<String, GGUFTensor> = FxHashMap::default();
        let arch = self
            .gguf_file
            .metadata
            .entries
            .get("general.architecture")
            .and_then(|value| match value {
                GGUFValue::String(s) => Some(s.as_str()),
                _ => None,
            });

        for tensor_info in &self.gguf_file.tensor_metadata {
            let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
            adjust_embedding_dims(&tensor_info.name, &mut dims, &self.gguf_file.metadata, arch);
            tensors.insert(
                tensor_info.name.clone(),
                GGUFTensor {
                    info: tensor_info.clone(),
                    dims: dims.clone(),
                },
            );
        }

        // Populate generic tensor infos
        let mut generic_tensor_infos = FxHashMap::default();
        for (name, tensor) in &tensors {
            let data_type = match Dtype::try_from(tensor.info.data_type) {
                Ok(dt) => dt,
                Err(_e) => {
                    panic!("Skipping tensor {} with unsupported type {:?}", name, tensor.info.data_type);
                }
            };

            generic_tensor_infos.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    dimensions: tensor.dims.clone(),
                    offset: tensor.info.offset,
                    data_type,
                },
            );
        }

        let meta_hint = layout_hint_from_metadata_with_key(&self.gguf_file.metadata, arch);
        let layout_hint = layout_hint_override()
            .or_else(|| meta_hint.as_ref().map(|(hint, _, _)| *hint))
            .unwrap_or_else(|| GGUFLayoutHint::from_architecture(arch));
        let debug_source = if layout_hint_override().is_some() {
            Some((METALLIC_GGUF_LAYOUT_OVERRIDE_ENV.to_string(), "<env override>".to_string()))
        } else {
            meta_hint.as_ref().map(|(_, key, value)| (key.clone(), value.clone()))
        };
        log_layout_hint(arch, layout_hint, debug_source);

        let mut metadata = self.gguf_file.metadata.clone();
        metadata.entries.insert(
            "metallic.gguf.layout_hint".to_string(),
            GGUFValue::String(match layout_hint {
                GGUFLayoutHint::Nk => "nk".to_string(),
                GGUFLayoutHint::Kn => "kn".to_string(),
            }),
        );

        Ok(GGUFModel {
            gguf_file: Arc::clone(&self.gguf_file),
            tensors,
            generic_tensor_infos,
            metadata,
        })
    }
}

/// Metadata describing a tensor stored in the GGUF file. Data is materialized on-demand.
pub struct GGUFTensor {
    info: GGUTensorInfo,
    dims: Vec<usize>,
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub gguf_file: Arc<GGUFFile>,
    pub tensors: FxHashMap<String, GGUFTensor>,
    pub generic_tensor_infos: FxHashMap<String, TensorInfo>,
    pub metadata: GGUFMetadata,
}

impl GGUFModel {
    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensor> {
        self.tensors.get(name)
    }

    /// Get model architecture from metadata
    pub fn get_architecture(&self) -> Option<&str> {
        if let Some(super::GGUFValue::String(arch)) = self.metadata.entries.get("general.architecture") {
            Some(arch)
        } else {
            None
        }
    }
}

// --- Implement ModelMetadata & LoadedModel ---

impl GGUFValue {
    fn to_metadata_value(&self) -> MetadataValue<'_> {
        match self {
            Self::U8(v) => MetadataValue::Int(i64::from(*v)),
            Self::I8(v) => MetadataValue::Int(i64::from(*v)),
            Self::U16(v) => MetadataValue::Int(i64::from(*v)),
            Self::I16(v) => MetadataValue::Int(i64::from(*v)),
            Self::U32(v) => MetadataValue::Int(i64::from(*v)),
            Self::I32(v) => MetadataValue::Int(i64::from(*v)),
            Self::F32(v) => MetadataValue::Float(f64::from(*v)),
            Self::Bool(v) => MetadataValue::Bool(*v),
            Self::String(v) => MetadataValue::String(Cow::Borrowed(v)),
            Self::Array(v) => MetadataValue::Array(v.iter().map(|item| item.to_metadata_value()).collect()),
            Self::U64(v) => MetadataValue::Int(*v as i64),
            Self::I64(v) => MetadataValue::Int(*v),
            Self::F64(v) => MetadataValue::Float(*v),
        }
    }
}

impl ModelMetadata for GGUFMetadata {
    fn get(&self, key: &str) -> Option<MetadataValue<'_>> {
        self.entries.get(key).map(|v| v.to_metadata_value())
    }

    fn parse_dtype(&self, s: &str) -> Option<Dtype> {
        match Dtype::parse_fuzzy(s) {
            Some(Dtype::F64) => Some(Dtype::F32), // Foundry downcasts F64 sources to F32.
            Some(Dtype::I32) => Some(Dtype::U32), // Preserve legacy metadata behavior.
            parsed => parsed,
        }
    }

    fn tokenizer_tokens(&self) -> Option<Vec<String>> {
        self.get_array_string("tokenizer.ggml.tokens")
    }

    fn tokenizer_merges(&self) -> Option<Vec<String>> {
        self.get_array_string("tokenizer.ggml.merges")
    }
}

impl LoadedModel for GGUFModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn architecture(&self) -> Option<&str> {
        self.get_architecture()
    }

    fn metadata(&self) -> &dyn ModelMetadata {
        &self.metadata
    }

    fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.generic_tensor_infos.get(name).cloned()
    }

    fn tensor_data(&self, name: &str) -> Result<TensorData<'_>, LoaderError> {
        let tensor = self.get_tensor(name).ok_or_else(|| LoaderError::TensorNotFound(name.to_string()))?;
        let data = self
            .gguf_file
            .get_tensor_data(&tensor.info)
            .map_err(|e| LoaderError::InvalidData(e.to_string()))?;
        Ok(TensorData::Slice(data))
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn estimated_memory_usage(&self) -> usize {
        self.gguf_file.mmap.len()
    }

    fn offload_tensor(&self, name: &str) -> Result<(), LoaderError> {
        let tensor = self.get_tensor(name).ok_or_else(|| LoaderError::TensorNotFound(name.to_string()))?;
        self.gguf_file
            .offload_tensor(&tensor.info)
            .map_err(|e| LoaderError::Io(std::io::Error::other(e)))
    }

    fn load_tensor(&self, name: &str) -> Result<(), LoaderError> {
        let tensor = self.get_tensor(name).ok_or_else(|| LoaderError::TensorNotFound(name.to_string()))?;
        let _ = self
            .gguf_file
            .load_tensor(&tensor.info)
            .map_err(|e| LoaderError::Io(std::io::Error::other(e)))?;
        Ok(())
    }

    fn available_fallbacks(&self) -> &[String] {
        static FALLBACKS: &[String] = &[];
        FALLBACKS
    }

    fn get_fallback(&self, _key: &str) -> Result<Option<TensorData<'_>>, LoaderError> {
        // GGUF loader currently doesn't implement implicit fallbacks.
        // If specific fallbacks are needed (e.g. legacy mapping), add them here.
        // For now, return None to indicate no fallback available for this key.
        Ok(None)
    }

    fn inferred_architecture_params(&self) -> Vec<(String, MetadataValue<'_>)> {
        let arch = self.architecture().or_else(|| metadata_architecture(&self.gguf_file.metadata));
        let mut params = Vec::new();

        // Common keys (many GGUFs expose these).
        if let Some(arch) = arch {
            let arch_vocab = format!("{arch}.vocab_size");
            if let Some(v) = self.metadata.get(&arch_vocab) {
                params.push(("vocab_size".to_string(), v));
            }
        }
        if !params.iter().any(|(name, _)| name == "vocab_size")
            && let Some(v) = self.metadata.get("model.vocab_size")
        {
            params.push(("vocab_size".to_string(), v));
        } else if !params.iter().any(|(name, _)| name == "vocab_size")
            && let Some(GGUFValue::Array(tokens)) = self.gguf_file.metadata.entries.get("tokenizer.ggml.tokens")
        {
            params.push(("vocab_size".to_string(), MetadataValue::Int(tokens.len() as i64)));
        }

        if let Some(arch) = arch {
            let mappings: [(&str, &[&str]); 8] = [
                ("d_model", &["embedding_length", "d_model", "hidden_size"]),
                ("n_heads", &["attention.head_count", "attention.n_heads", "n_head"]),
                ("n_kv_heads", &["attention.head_count_kv", "attention.n_kv_heads", "n_head_kv"]),
                ("n_layers", &["block_count", "n_layer", "num_hidden_layers"]),
                ("ff_dim", &["feed_forward_length", "ffn_dim", "intermediate_size"]),
                ("max_seq_len", &["context_length", "max_context_length", "max_sequence_length"]),
                ("rope_base", &["rope.freq_base", "rope.theta", "rope_freq_base"]),
                (
                    "rms_eps",
                    &["attention.layer_norm_rms_epsilon", "layer_norm_rms_epsilon", "rms_norm_eps"],
                ),
            ];

            for (field, suffixes) in mappings {
                for suffix in suffixes {
                    let key = format!("{arch}.{suffix}");
                    if let Some(v) = self.metadata.get(&key) {
                        params.push((field.to_string(), v));
                        break;
                    }
                }
            }
        }

        params
    }
}

#[path = "model_loader.test.rs"]
mod tests;
