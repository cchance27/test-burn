//! Embedding Lookup Kernel for Foundry.
//!
//! Gathers rows from an embedding table based on token indices.
//! Each output element is copied from table[indices[pos], feat].

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{spec::DynamicValue, types::TensorArg};

pub mod step;

/// Parameters for embedding lookup kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct EmbeddingParams {
    /// Feature dimension (d_model).
    pub d_model: DynamicValue<u32>,
    /// Total output elements (batch * seq * d_model).
    pub total_elements: DynamicValue<u32>,
    /// Vocabulary size for bounds checking.
    pub vocab_size: DynamicValue<u32>,
}

/// Embedding kernel definition for Step derivation.
/// Actual execution delegates to dynamic CompoundKernels via manual `execute`.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "embedding/embedding.metal", // Placeholder, not used by manual execute
    function = "embedding_kernel",        // Placeholder
    args = EmbeddingParams,               // Unresolved params for Step
    step = true,
    execute = false
)]
pub struct Embedding {
    pub table: TensorArg,
    #[arg(scale_for = "table")]
    pub scale_bytes: TensorArg, // Derived from table name ("{table}_scales"), populated as index in CompiledStep
    pub indices: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: EmbeddingParamsResolved,
}

#[path = "mod.test.rs"]
mod tests;
