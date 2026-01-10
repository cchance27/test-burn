//! Embedding Lookup Kernel for Foundry.
//!
//! Gathers rows from an embedding table based on token indices.
//! Each output element is copied from table[indices[pos], feat].

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{foundry::spec::DynamicValue, types::TensorArg};

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

/// Embedding lookup kernel.
///
/// Gathers embedding vectors for given token indices.
/// - table: [vocab_size, d_model] embedding matrix
/// - indices: [batch * seq] token ids
/// - output: [batch * seq * d_model] gathered embeddings
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "embedding/embedding.metal",
    function = "embedding_lookup_f16",
    args = EmbeddingParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = false
)]
pub struct Embedding {
    /// Embedding table buffer.
    pub table: TensorArg,
    /// Token indices buffer.
    pub indices: TensorArg,
    /// Output buffer.
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: EmbeddingParamsResolved,
}

impl Embedding {
    /// Create a new embedding lookup kernel.
    pub fn new(table: &TensorArg, indices: &TensorArg, output: &TensorArg, params: EmbeddingParamsResolved) -> Self {
        Self {
            table: table.clone(),
            indices: indices.clone(),
            output: output.clone(),
            params,
        }
    }
}

/// Embedding lookup kernel for Q8_0 tables (split data + scales).
///
/// - `table` is int8 weights as bytes (one byte per weight)
/// - `scale_bytes` stores fp16 scales (2 bytes) per 32-weight block, row-major by token row.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "embedding/embedding.metal",
    function = "embedding_lookup_q8",
    args = EmbeddingParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = false
)]
pub struct EmbeddingQ8 {
    pub table: TensorArg,
    pub scale_bytes: TensorArg,
    pub indices: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: EmbeddingParamsResolved,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_params_metal_struct() {
        let def = EmbeddingParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct EmbeddingParams"));
        assert!(def.contains("d_model"));
        assert!(def.contains("total_elements"));
        assert!(def.contains("vocab_size"));
    }
}
