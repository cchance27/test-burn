//! Typestate ModelBuilder for safe model loading.
//!
//! Enforces correct loading order at compile time:
//! `Empty` → `WithSpec` → `WithWeights` → `CompiledModel`

use std::{marker::PhantomData, path::Path};

use super::executor::CompiledModel;
use crate::{
    Foundry, error::MetalError, gguf::{
        GGUFFile, model_loader::{GGUFModel, GGUFModelLoader}
    }, spec::ModelSpec
};

// =============================================================================
// Typestate Markers
// =============================================================================

/// Initial state: no spec or weights loaded.
pub struct Empty;

/// Spec loaded, awaiting weights.
pub struct WithSpec;

/// Spec and weights loaded, ready to compile.
pub struct WithWeights;

// =============================================================================
// Weight Bundle
// =============================================================================

/// Contains loaded weights from a GGUF file.
///
/// Tensor types (F16, Q8, etc.) are determined per-tensor at materialization time,
/// and policy selection uses `T::Policy` from the layer's `TensorElement` type.
pub struct WeightBundle {
    /// The loaded GGUF model with tensors and metadata
    model: GGUFModel,
}

impl WeightBundle {
    /// Get a reference to the underlying GGUFModel
    pub fn gguf_model(&self) -> &GGUFModel {
        &self.model
    }

    /// Get the architecture string (e.g., "qwen2", "llama")
    pub fn architecture(&self) -> Option<&str> {
        self.model.get_architecture()
    }

    /// Get a specific tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&crate::gguf::model_loader::GGUFTensor> {
        self.model.get_tensor(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.model.tensors.keys()
    }

    #[cfg(test)]
    pub(crate) fn new_empty() -> Self {
        Self {
            model: crate::gguf::model_loader::GGUFModel::new_empty(),
        }
    }
}

// =============================================================================
// ModelBuilder
// =============================================================================

/// Builder that enforces model loading order at compile time via typestate.
///
/// # Example
/// ```rust,ignore
/// let model = ModelBuilder::new()
///     .with_spec_file("models/qwen25.json")?
///     .with_gguf("models/qwen25-q8.gguf")?
///     .build(&mut foundry)?;
/// ```
pub struct ModelBuilder<State = Empty> {
    spec: Option<ModelSpec>,
    weights: Option<WeightBundle>,
    _state: PhantomData<State>,
}

impl ModelBuilder<Empty> {
    /// Create a new empty ModelBuilder.
    pub fn new() -> Self {
        Self {
            spec: None,
            weights: None,
            _state: PhantomData,
        }
    }

    /// Load execution spec from a ModelSpec struct.
    pub fn with_spec(self, spec: ModelSpec) -> ModelBuilder<WithSpec> {
        ModelBuilder {
            spec: Some(spec),
            weights: None,
            _state: PhantomData,
        }
    }

    /// Load execution spec from a JSON file.
    pub fn with_spec_file(self, path: impl AsRef<Path>) -> Result<ModelBuilder<WithSpec>, MetalError> {
        let spec = ModelSpec::from_file(path).map_err(|e| MetalError::InvalidShape(format!("Failed to load spec: {}", e)))?;
        Ok(self.with_spec(spec))
    }
}

impl Default for ModelBuilder<Empty> {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder<WithSpec> {
    /// Load weights from a GGUF file.
    pub fn with_gguf(self, path: impl AsRef<Path>) -> Result<ModelBuilder<WithWeights>, MetalError> {
        let path = path.as_ref();

        let gguf_file =
            GGUFFile::load_mmap_and_get_metadata(path).map_err(|e| MetalError::InvalidShape(format!("Failed to load GGUF: {}", e)))?;

        let loader = GGUFModelLoader::new(gguf_file);
        let model = loader
            .load_model()
            .map_err(|e| MetalError::InvalidShape(format!("Failed to parse GGUF model: {}", e)))?;

        Ok(ModelBuilder {
            spec: self.spec,
            weights: Some(WeightBundle { model }),
            _state: PhantomData,
        })
    }
}

impl ModelBuilder<WithWeights> {
    /// Get information about the loaded weights.
    pub fn weights_info(&self) -> Option<&WeightBundle> {
        self.weights.as_ref()
    }

    /// Compile the model into an executable form.
    pub fn build(self, foundry: &mut Foundry) -> Result<CompiledModel, MetalError> {
        let mut spec = self
            .spec
            .ok_or_else(|| MetalError::InvalidShape("ModelBuilder: spec not loaded".to_string()))?;
        let weights = self
            .weights
            .ok_or_else(|| MetalError::InvalidShape("ModelBuilder: weights not loaded".to_string()))?;

        // Baseline architecture comes from GGUF metadata; DSL can override, runtime can override later.
        //
        // If the spec does not provide `architecture.metadata_keys`, fall back to built-in mappings
        // for common architectures (DEBT: keep this limited; prefer explicit keys in the spec).
        let metadata = weights.gguf_model().get_metadata();
        let defaults = if spec.architecture.metadata_keys.keys.is_empty() {
            super::metadata_defaults::infer_architecture_defaults_from_gguf_metadata(metadata)?
        } else {
            super::metadata_defaults::infer_from_gguf_with_keys(metadata, &spec.architecture.metadata_keys)?
        };
        spec.architecture.apply_metadata_baseline(&defaults)?;

        let model = CompiledModel::new(spec, weights)?;

        // Prepare a reusable execution session during model load so generation doesn't pay the one-time
        // weight materialization / buffer allocation cost. This aligns Foundry's "cold start" behavior
        // with Context (heavy work in model load, cheap per-generation setup).
        model.initialize_session(foundry)?;

        Ok(model)
    }
}
