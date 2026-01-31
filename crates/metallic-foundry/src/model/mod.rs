//! Foundry Model Module
//!
//! Provides typestate-based model loading and compiled model execution.

mod builder;
mod context;
mod executor;
mod metadata_defaults;

pub use builder::ModelBuilder;
pub use context::{ContextConfig, EvictionPolicy};
pub use executor::CompiledModel;
pub use metadata_defaults::infer_from_gguf as infer_architecture_defaults_from_gguf_metadata;
