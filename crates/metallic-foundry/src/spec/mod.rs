//! Foundry Execution Spec Module
//!
//! Provides the Step trait and ModelSpec for declarative model execution.

mod dynamic;
mod int_expr;
mod model_spec;
mod repeat;
mod step;

pub use dynamic::{DynamicValue, Resolvable};
pub use int_expr::IntExpr;
pub use model_spec::{
    Architecture, ArchitectureDefaults, LayerTensorNames, MetadataKeysSpec, MetadataValue, ModelSpec, StorageClass, TensorAllocSpec, TensorNames, WeightBindingSpec, WeightLayoutSpec
};
pub use repeat::Repeat;
pub use step::{Ref, Step, TensorBindings};

pub mod compiled;
pub use compiled::{CompiledStep, FastBindings, ResolvedSymbols, SymbolTable};
