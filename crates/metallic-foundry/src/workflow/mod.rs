//! Workflow system for composing multi-model inference pipelines.
//!
//! Workflows are loaded from JSON and executed against a set of named models.

mod compiler;
pub mod ops;
mod runner;
mod spec;
mod value;

pub use runner::{WorkflowRunner, WorkflowRunnerConfig};
pub use spec::{Param, WorkflowModelResourceSpec, WorkflowResourcesSpec, WorkflowSpec, WorkflowStepSpec};
pub use value::Value;

#[cfg(test)]
mod tests;
