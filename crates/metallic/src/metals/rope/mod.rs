//! RoPE (Rotary Position Embedding) Kernel for Foundry.
//!
//! Applies rotary position embeddings to input tensors using precomputed cos/sin caches.
//! Each pair of features (i, i+half_dim) is rotated: out_i = x_i*cos - x_j*sin, out_j = x_j*cos + x_i*sin

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for RoPE kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct RopeParams {
    /// Feature dimension (must be even).
    pub dim: u32,
    /// Sequence length.
    pub seq_len: u32,
    /// Position offset for incremental decoding.
    pub position_offset: u32,
    /// Total elements in input tensor.
    pub total_elements: u32,
}

/// RoPE (Rotary Position Embedding) kernel.
///
/// Applies rotation to paired features using cos/sin caches.
#[derive(KernelArgs, Clone)]
pub struct Rope {
    /// Input tensor.
    #[arg(buffer = 0)]
    pub input: TensorArg,
    /// Output tensor (same shape as input).
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    /// Precomputed cosine cache [max_seq, dim/2].
    #[arg(buffer = 2)]
    pub cos: TensorArg,
    /// Precomputed sine cache [max_seq, dim/2].
    #[arg(buffer = 3)]
    pub sin: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 4)]
    pub params: RopeParams,
}

impl Rope {
    /// Create a new RoPE kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, cos: &TensorArg, sin: &TensorArg, params: RopeParams) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            params,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct RopeId;

impl Kernel for Rope {
    type Args = RopeParams;
    type Id = RopeId;

    fn source(&self) -> KernelSource {
        KernelSource::File("rope/rope.metal")
    }

    fn function_name(&self) -> &'static str {
        "rope_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        RopeParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // Thread-based: one thread per element
        let total = self.params.total_elements as usize;
        let threads_per_group = 256;
        let num_groups = (total + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("RoPE kernel does not yet support compound kernel staging - needs Metal template refactoring")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_params_metal_struct() {
        let def = RopeParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct RopeParams"));
        assert!(def.contains("dim"));
        assert!(def.contains("seq_len"));
        assert!(def.contains("position_offset"));
        assert!(def.contains("total_elements"));
    }
}
