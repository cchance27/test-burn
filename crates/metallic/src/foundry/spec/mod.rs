//! Foundry Execution Spec Module
//!
//! Provides the Step trait and ModelSpec for declarative model execution.

mod dynamic;
mod model_spec;
mod repeat;
mod step;

pub use dynamic::{DynamicValue, Resolvable};
pub use model_spec::{Architecture, LayerTensorNames, ModelSpec, TensorNames};
pub use repeat::Repeat;
pub use step::{Ref, Step, TensorBindings};

pub mod compiled;
pub use compiled::{CompiledStep, FastBindings, SymbolTable};
