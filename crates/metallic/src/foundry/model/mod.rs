//! Foundry Model Module
//!
//! Provides typestate-based model loading and compiled model execution.

mod builder;
mod executor;

pub use builder::ModelBuilder;
pub use executor::CompiledModel;
