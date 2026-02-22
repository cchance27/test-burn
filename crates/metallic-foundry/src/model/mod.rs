//! Foundry Model Module
//!
//! Provides typestate-based model loading and compiled model execution.

mod builder;
mod context;
mod executor;
mod kv_geometry;
mod metadata_defaults;

pub use builder::ModelBuilder;
pub use context::{ContextConfig, EvictionPolicy};
pub use executor::CompiledModel;
pub use kv_geometry::{KvCacheLayout, KvGeometry};
pub use metadata_defaults::infer_architecture_defaults;
