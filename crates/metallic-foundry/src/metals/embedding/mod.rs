//! Embedding Lookup Kernel for Foundry.
//!
//! Gathers rows from an embedding table based on token indices.
//! Each output element is copied from table[indices[pos], feat].

use metallic_macros::MetalStruct;

use crate::spec::DynamicValue;

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

// NOTE: Legacy static kernel structs have been removed.
// All embedding lookups now go through EmbeddingStep which uses CompoundKernel
// with dynamic policy selection based on tensor dtype.
//
// The pattern is:
//   EmbeddingStep -> EmbeddingStage -> CompoundKernel::compile()
//   -> includes policy header first, then embedding.metal template
//   -> emits: run_embedding_core<{Policy}>(...)

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
