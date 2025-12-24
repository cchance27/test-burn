//! Embedding Lookup Kernel for Foundry.
//!
//! Gathers rows from an embedding table based on token indices.
//! Each output element is copied from table[indices[pos], feat].

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for embedding lookup kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct EmbeddingParams {
    /// Feature dimension (d_model).
    pub d_model: u32,
    /// Total output elements (batch * seq * d_model).
    pub total_elements: u32,
    /// Vocabulary size for bounds checking.
    pub vocab_size: u32,
}

/// Embedding lookup kernel.
///
/// Gathers embedding vectors for given token indices.
/// - table: [vocab_size, d_model] embedding matrix
/// - indices: [batch * seq] token ids
/// - output: [batch * seq * d_model] gathered embeddings
#[derive(KernelArgs, Clone)]
pub struct Embedding {
    /// Embedding table buffer.
    #[arg(buffer = 0)]
    pub table: TensorArg,
    /// Token indices buffer.
    #[arg(buffer = 1)]
    pub indices: TensorArg,
    /// Output buffer.
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 3)]
    pub params: EmbeddingParams,
}

impl Embedding {
    /// Create a new embedding lookup kernel.
    pub fn new(table: &TensorArg, indices: &TensorArg, output: &TensorArg, params: EmbeddingParams) -> Self {
        Self {
            table: table.clone(),
            indices: indices.clone(),
            output: output.clone(),
            params,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct EmbeddingId;

impl Kernel for Embedding {
    type Args = EmbeddingParams;
    type Id = EmbeddingId;

    fn source(&self) -> KernelSource {
        KernelSource::File("embedding/embedding.metal")
    }

    fn function_name(&self) -> &'static str {
        "embedding_lookup_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        EmbeddingParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // Thread-based: one thread per output element
        let total = self.params.total_elements as usize;
        let threads_per_group = 256;
        let num_groups = (total + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("Embedding kernel does not yet support compound kernel staging - needs Metal template refactoring")
    }
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
