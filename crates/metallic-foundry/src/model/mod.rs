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
