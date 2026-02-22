// Re-export SDK types
pub use metallic_sdk::{
    model::{LoadedModel, LoaderError, MetadataValue, ModelMetadata, TensorData, TensorInfo}, tensor::Dtype
};

pub mod quant_spec;

#[cfg(feature = "gguf")]
pub(crate) mod gguf;

mod loader;
pub use loader::ModelLoader;
use rustc_hash::FxHashSet;

/// Loader-side tensor-name index for fast, format-agnostic binding resolution.
///
/// Foundry/runtime can use this to resolve logical tensor candidates without
/// coupling directly to source-format-specific internals (GGUF today, others later).
#[derive(Debug, Clone, Default)]
pub struct TensorNameIndex {
    names: FxHashSet<String>,
}

impl TensorNameIndex {
    /// Build an index from a loaded model's exposed tensor names.
    pub fn from_model(model: &dyn LoadedModel) -> Self {
        let names = model.tensor_names().into_iter().collect();
        Self { names }
    }

    /// Returns true if a tensor name exists in the loaded model.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// Return the first candidate tensor name that exists in this model.
    pub fn resolve_first<'a, I>(&self, candidates: I) -> Option<String>
    where
        I: IntoIterator<Item = &'a str>,
    {
        for candidate in candidates {
            if self.contains(candidate) {
                return Some(candidate.to_string());
            }
        }
        None
    }
}

// --- Testing Utilities ---

pub struct DummyMetadata;
impl ModelMetadata for DummyMetadata {
    fn get(&self, _key: &str) -> Option<MetadataValue<'_>> {
        None
    }
    fn parse_dtype(&self, _s: &str) -> Option<Dtype> {
        None
    }
    fn tokenizer_tokens(&self) -> Option<Vec<String>> {
        None
    }
    fn tokenizer_merges(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct DummyModel;
impl LoadedModel for DummyModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn architecture(&self) -> Option<&str> {
        None
    }
    fn metadata(&self) -> &dyn ModelMetadata {
        &DummyMetadata
    }
    fn tensor_info(&self, _name: &str) -> Option<TensorInfo> {
        None
    }
    fn tensor_data(&self, _name: &str) -> Result<TensorData<'_>, LoaderError> {
        Err(LoaderError::TensorNotFound(_name.to_string()))
    }
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn estimated_memory_usage(&self) -> usize {
        0
    }
    fn offload_tensor(&self, _name: &str) -> Result<(), LoaderError> {
        Ok(())
    }
    fn load_tensor(&self, _name: &str) -> Result<(), LoaderError> {
        Ok(())
    }
    fn available_fallbacks(&self) -> &[String] {
        &[]
    }
    fn get_fallback(&self, _key: &str) -> Result<Option<TensorData<'_>>, LoaderError> {
        Ok(None)
    }
    fn inferred_architecture_params(&self) -> Vec<(String, MetadataValue<'_>)> {
        Vec::new()
    }
}

pub struct MapMetadata {
    pub entries: rustc_hash::FxHashMap<String, MetadataValue<'static>>,
}
impl ModelMetadata for MapMetadata {
    fn get(&self, key: &str) -> Option<MetadataValue<'_>> {
        self.entries.get(key).cloned()
    }
    fn parse_dtype(&self, _s: &str) -> Option<Dtype> {
        None
    }
    fn tokenizer_tokens(&self) -> Option<Vec<String>> {
        None
    }
    fn tokenizer_merges(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct MockModel {
    pub architecture: Option<String>,
    pub metadata: MapMetadata,
    pub inferred_params: Vec<(String, MetadataValue<'static>)>,
}
impl LoadedModel for MockModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }
    fn metadata(&self) -> &dyn ModelMetadata {
        &self.metadata
    }
    fn tensor_info(&self, _name: &str) -> Option<TensorInfo> {
        None
    }
    fn tensor_data(&self, _name: &str) -> Result<TensorData<'_>, LoaderError> {
        Err(LoaderError::TensorNotFound(_name.to_string()))
    }
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn estimated_memory_usage(&self) -> usize {
        0
    }
    fn offload_tensor(&self, _name: &str) -> Result<(), LoaderError> {
        Ok(())
    }
    fn load_tensor(&self, _name: &str) -> Result<(), LoaderError> {
        Ok(())
    }
    fn available_fallbacks(&self) -> &[String] {
        &[]
    }
    fn get_fallback(&self, _key: &str) -> Result<Option<TensorData<'_>>, LoaderError> {
        Ok(None)
    }
    fn inferred_architecture_params(&self) -> Vec<(String, MetadataValue<'_>)> {
        self.inferred_params.clone()
    }
}
