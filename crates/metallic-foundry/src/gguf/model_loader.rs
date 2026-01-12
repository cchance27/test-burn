use std::sync::Arc;

use objc2_metal::{MTLBuffer, MTLDevice};
use rustc_hash::FxHashMap;

use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::gguf::{
    GGUFValue, file::GGUFMetadata, tensor_info::{GGUFRawTensor, GGUTensorInfo}
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGUFLayoutHint {
    Nk,
    Kn,
}

impl GGUFLayoutHint {
    fn from_architecture(arch: Option<&str>) -> Self {
        match arch.map(|s| s.to_ascii_lowercase()) {
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
    std::env::var(METALLIC_GGUF_LAYOUT_DEBUG_ENV)
        .ok()
        .map(|v| v.trim() != "0")
        .unwrap_or(false)
}

fn log_layout_hint(arch: Option<&str>, hint: GGUFLayoutHint, source: Option<(String, String)>) {
    if !layout_debug_enabled() {
        return;
    }
    match source {
        Some((key, value)) => {
            eprintln!(
                "[GGUF_LAYOUT] arch={:?} hint={:?} source_key={} source_value={}",
                arch, hint, key, value
            );
        }
        None => {
            eprintln!("[GGUF_LAYOUT] arch={:?} hint={:?} source_key=<none>", arch, hint);
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

fn metadata_vocab_size(metadata: &GGUFMetadata) -> Option<usize> {
    if let Some(v) = metadata_usize(metadata, &["vocab_size", "model.vocab_size"]) {
        return Some(v);
    }

    if let Some(GGUFValue::Array(values)) = metadata.entries.get("tokenizer.ggml.tokens") {
        return Some(values.len());
    }

    None
}

fn adjust_embedding_dims(_name: &str, dims: &mut [usize], metadata: &GGUFMetadata) {
    if dims.len() != 2 {
        return;
    }

    let d_model = metadata_usize(
        metadata,
        &[
            "qwen2.embedding_length",
            "qwen2.d_model",
            "model.d_model",
            "llama.embedding_length",
            "llama.d_model",
        ],
    );
    let vocab = metadata_vocab_size(metadata);

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

    /// Return the size of the memory-mapped GGUF file in bytes so callers can
    /// reason about the resident host footprint of keeping the loader alive.
    pub fn mapped_len(&self) -> usize {
        self.gguf_file.mmap.len()
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
            adjust_embedding_dims(&tensor_info.name, &mut dims, &self.gguf_file.metadata);
            let expected_elements: usize = dims.iter().product();

            tensors.insert(
                tensor_info.name.clone(),
                GGUFTensor {
                    info: tensor_info.clone(),
                    dims,
                    expected_elements,
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

        Ok(GGUFModel {
            gguf_file: Arc::clone(&self.gguf_file),
            tensors,
            metadata: self.gguf_file.metadata.clone(),
            layout_hint,
        })
    }
}

impl GGUFModel {
    /// Helper to map GGUF data types to Metallic Dtype.
    pub fn map_dtype(gguf_type: GGUFDataType) -> crate::tensor::Dtype {
        match gguf_type {
            GGUFDataType::F32 => crate::tensor::Dtype::F32,
            GGUFDataType::F16 => crate::tensor::Dtype::F16,
            GGUFDataType::Q8_0 => crate::tensor::Dtype::U8,
            GGUFDataType::Q8_1 => crate::tensor::Dtype::U8,
            _ => crate::tensor::Dtype::F16, // Default fallback
        }
    }
}

/// Metadata describing a tensor stored in the GGUF file. Data is materialized on-demand.
pub struct GGUFTensor {
    info: GGUTensorInfo,
    dims: Vec<usize>,
    expected_elements: usize,
}

impl GGUFTensor {
    pub fn data_type(&self) -> GGUFDataType {
        self.info.data_type
    }

    pub fn is_empty(&self) -> bool {
        self.expected_elements == 0
    }

    pub fn len(&self) -> usize {
        self.expected_elements
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Materialize the tensor directly to a Metal buffer for use in Foundry.
    pub fn materialize_to_device(
        &self,
        file: &GGUFFile,
        device: &crate::types::MetalDevice,
    ) -> Result<crate::types::MetalBuffer, GGUFError> {
        let view = self.info.view(file)?;
        let raw_bytes = match view {
            GGUFRawTensor::F32(slice) => bytemuck::cast_slice(slice),
            GGUFRawTensor::F16(slice) => bytemuck::cast_slice(slice),
            GGUFRawTensor::Bytes(bytes, _) => bytes,
        };

        let buffer = device
            .0
            .newBufferWithLength_options(raw_bytes.len(), objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Failed to allocate Metal buffer for tensor '{}'", self.info.name)))?;

        unsafe {
            let ptr = buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(raw_bytes.as_ptr(), ptr, raw_bytes.len());
        }

        Ok(crate::types::MetalBuffer(buffer))
    }

    pub fn raw_view<'a>(&self, file: &'a GGUFFile) -> Result<GGUFRawTensor<'a>, GGUFError> {
        self.info.view(file)
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub gguf_file: Arc<GGUFFile>,
    pub tensors: FxHashMap<String, GGUFTensor>,
    pub metadata: GGUFMetadata,
    pub layout_hint: GGUFLayoutHint,
}

macro_rules! create_metadata_getter {
    ($func_name:ident, $variant:path, $return_type:ty) => {
        pub fn $func_name(&self, names: &[&str], or: $return_type) -> $return_type {
            for name in names {
                if let Some($variant(v)) = self.metadata.entries.get(*name) {
                    return *v; // Return the found value
                }
            }
            or // Return the default value if nothing was found
        }
    };
}

impl GGUFModel {
    create_metadata_getter!(get_metadata_u32_or, GGUFValue::U32, u32);
    create_metadata_getter!(get_metadata_f32_or, GGUFValue::F32, f32);

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensor> {
        self.tensors.get(name)
    }

    pub fn tensor_raw_view(&self, name: &str) -> Result<GGUFRawTensor<'_>, GGUFError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Tensor '{}' not found in GGUF metadata", name)))?;
        tensor.raw_view(&self.gguf_file)
    }

    /// Get model metadata
    pub fn get_metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    /// Get model architecture from metadata
    pub fn get_architecture(&self) -> Option<&str> {
        if let Some(super::GGUFValue::String(arch)) = self.metadata.entries.get("general.architecture") {
            Some(arch)
        } else {
            None
        }
    }

    pub fn layout_hint(&self) -> GGUFLayoutHint {
        self.layout_hint
    }

    /// Get context length from metadata
    pub fn get_context_length(&self) -> Option<u64> {
        if let Some(super::GGUFValue::U32(len)) = self.metadata.entries.get("qwen2.context_length") {
            Some(*len as u64)
        } else {
            None
        }
    }
}
