//! RoPE (Rotary Position Embedding) Kernel for Foundry.
//!
//! Applies rotary position embeddings to input tensors using precomputed cos/sin caches.
//! Each pair of features (i, i+half_dim) is rotated: out_i = x_i*cos - x_j*sin, out_j = x_j*cos + x_i*sin

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{foundry::spec::DynamicValue, types::TensorArg};

/// Parameters for RoPE kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct RopeParams {
    /// Feature dimension (must be even).
    pub dim: u32,
    /// Sequence length.
    pub seq_len: DynamicValue<u32>,
    /// Position offset for incremental decoding (can be dynamic).
    pub position_offset: DynamicValue<u32>,
    /// Total elements in input tensor.
    pub total_elements: DynamicValue<u32>,
}

/// RoPE (Rotary Position Embedding) kernel.
///
/// Applies rotation to paired features using cos/sin caches.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "rope/rope.metal",
    function = "rope_kernel_f16",
    args = RopeParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = true
)]
pub struct Rope {
    /// Input tensor.
    pub input: TensorArg,
    /// Output tensor (same shape as input).
    #[arg(output)]
    pub output: TensorArg,
    /// Precomputed cosine cache [max_seq, dim/2].
    pub cos: TensorArg,
    /// Precomputed sine cache [max_seq, dim/2].
    pub sin: TensorArg,
    /// Kernel parameters (resolved from dynamic values).
    pub params: RopeParamsResolved,
}

impl Rope {
    /// Create a new RoPE kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, cos: &TensorArg, sin: &TensorArg, params: RopeParamsResolved) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            params,
        }
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
pub mod stage;
