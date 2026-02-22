//! Typestate ModelBuilder for safe model loading.
//!
//! Enforces correct loading order at compile time:
//! `Empty` → `WithSpec` → `WithWeights` → `CompiledModel`

use std::{marker::PhantomData, path::Path};

use metallic_loader::LoadedModel;

use super::executor::CompiledModel;
use crate::{Foundry, error::MetalError, spec::ModelSpec};

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

pub struct WeightBundle {
    /// The loaded model with tensors and metadata
    model: Box<dyn LoadedModel>,
}

impl WeightBundle {
    /// Get a reference to the underlying LoadedModel
    pub fn model(&self) -> &dyn LoadedModel {
        self.model.as_ref()
    }

    /// Get the architecture string (e.g., "qwen2", "llama")
    pub fn architecture(&self) -> Option<&str> {
        self.model.architecture()
    }

    /// Get a specific tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<metallic_loader::TensorInfo> {
        self.model.tensor_info(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.model.tensor_names()
    }

    #[cfg(test)]
    pub(crate) fn new_empty() -> Self {
        Self {
            model: Box::new(metallic_loader::DummyModel),
        }
    }
}

// =============================================================================
// ModelBuilder
// =============================================================================

/// Builder that enforces model loading order at compile time via typestate.
///
/// # Example
/// ```text
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
    /// Load weights from an abstract LoadedModel.
    pub fn with_model(self, model: Box<dyn LoadedModel>) -> ModelBuilder<WithWeights> {
        ModelBuilder {
            spec: self.spec,
            weights: Some(WeightBundle { model }),
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<WithWeights> {
    /// Get information about the loaded weights.
    pub fn weights_info(&self) -> Option<&WeightBundle> {
        self.weights.as_ref()
    }

    fn compile_model(self) -> Result<CompiledModel, MetalError> {
        let mut spec = self
            .spec
            .ok_or_else(|| MetalError::InvalidShape("ModelBuilder: spec not loaded".to_string()))?;
        let weights = self
            .weights
            .ok_or_else(|| MetalError::InvalidShape("ModelBuilder: weights not loaded".to_string()))?;

        // Baseline architecture comes from model metadata; DSL can override, runtime can override later.
        //
        // If the spec does not provide `architecture.metadata_keys`, fall back to built-in mappings
        // for common architectures (DEBT: keep this limited; prefer explicit keys in the spec).
        let metadata = weights.model().metadata();
        let defaults = if spec.architecture.metadata_keys.keys.is_empty() {
            super::metadata_defaults::infer_architecture_defaults(weights.model())?
        } else {
            super::metadata_defaults::infer_from_metadata_with_keys(metadata, &spec.architecture.metadata_keys)?
        };
        spec.architecture.apply_metadata_baseline(&defaults)?;

        CompiledModel::new(spec, weights)
    }

    /// Compile the model and eagerly initialize a reusable execution session.
    pub fn build(self, foundry: &mut Foundry) -> Result<CompiledModel, MetalError> {
        let model = self.compile_model()?;
        // Prepare a reusable execution session during model load so generation doesn't pay the one-time
        // weight materialization / buffer allocation cost. This aligns Foundry's "cold start" behavior
        // with Context (heavy work in model load, cheap per-generation setup).
        model.initialize_session(foundry)?;
        Ok(model)
    }

    /// Compile the model without eager session initialization.
    ///
    /// Use this when model load latency is more important than first-inference latency.
    pub fn build_lazy(self) -> Result<CompiledModel, MetalError> {
        self.compile_model()
    }
}
