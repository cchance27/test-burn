//! RepeatKvHeads Kernel for Foundry.
//!
//! Repeats K/V heads for Grouped Query Attention (GQA).
//! Input: [batch * n_kv_heads, cache_stride, head_dim]
//! Output: [batch * n_heads, seq, head_dim]

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for RepeatKvHeads kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct RepeatKvHeadsParams {
    /// Group size (n_heads / n_kv_heads).
    pub group_size: u32,
    /// Batch size.
    pub batch: u32,
    /// Number of KV heads.
    pub n_kv_heads: u32,
    /// Number of query heads.
    pub n_heads: u32,
    /// Sequence length.
    pub seq: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Cache stride (max sequence capacity).
    pub cache_stride: u32,
    /// Total output elements.
    pub total_elements: u32,
}

/// RepeatKvHeads kernel.
///
/// Repeats K/V from n_kv_heads → n_heads for attention.
#[derive(KernelArgs, Clone)]
pub struct RepeatKvHeads {
    /// Input tensor [batch * n_kv_heads, cache_stride, head_dim].
    #[arg(buffer = 0)]
    pub input: TensorArg,
    /// Output tensor [batch * n_heads, seq, head_dim].
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 2)]
    pub params: RepeatKvHeadsParams,
}

impl RepeatKvHeads {
    /// Create a new RepeatKvHeads kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, params: RepeatKvHeadsParams) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            params,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct RepeatKvHeadsId;

impl Kernel for RepeatKvHeads {
    type Args = RepeatKvHeadsParams;
    type Id = RepeatKvHeadsId;

    fn source(&self) -> KernelSource {
        KernelSource::File("repeat_kv_heads/repeat_kv_heads.metal")
    }

    fn function_name(&self) -> &'static str {
        "repeat_kv_heads_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        RepeatKvHeadsParams::METAL_STRUCT_DEF.to_string()
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
        todo!("RepeatKvHeads kernel does not yet support compound kernel staging")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_kv_heads_params_metal_struct() {
        let def = RepeatKvHeadsParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct RepeatKvHeadsParams"));
        assert!(def.contains("group_size"));
        assert!(def.contains("n_heads"));
        assert!(def.contains("cache_stride"));
    }
}
