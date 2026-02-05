// Re-export SDK types
pub use metallic_sdk::{
    model::{LoadedModel, LoaderError, MetadataValue, ModelMetadata, TensorData, TensorInfo}, tensor::Dtype
};

#[cfg(feature = "gguf")]
pub(crate) mod gguf;

mod loader;
pub use loader::ModelLoader;

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
