//! KV Rearrange Kernel for Foundry.
//!
//! Rearranges QKV outputs from [batch*seq, kv_dim] layout to [batch*n_heads, seq, head_dim]
//! for attention computation. Handles GQA (Grouped Query Attention) via group_size.

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for KV Rearrange kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct KvRearrangeParams {
    /// KV dimension (total features in K or V).
    pub kv_dim: u32,
    /// Row stride in input tensor.
    pub row_stride: u32,
    /// Dimension per KV head.
    pub kv_head_dim: u32,
    /// Number of query heads.
    pub n_heads: u32,
    /// Number of KV heads (for GQA).
    pub n_kv_heads: u32,
    /// Dimension per output head.
    pub head_dim: u32,
    /// Sequence length.
    pub seq: u32,
    /// Total output elements.
    pub total_elements: u32,
}

/// KV Rearrange kernel.
///
/// Rearranges [batch*seq, kv_dim] → [batch*n_heads, seq, head_dim].
#[derive(KernelArgs, Clone)]
pub struct KvRearrange {
    /// Input tensor [batch*seq, kv_dim].
    #[arg(buffer = 0)]
    pub input: TensorArg,
    /// Output tensor [batch*n_heads, seq, head_dim].
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 2)]
    pub params: KvRearrangeParams,
}

impl KvRearrange {
    /// Create a new KV Rearrange kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, params: KvRearrangeParams) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            params,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct KvRearrangeId;

impl Kernel for KvRearrange {
    type Args = KvRearrangeParams;
    type Id = KvRearrangeId;

    fn source(&self) -> KernelSource {
        KernelSource::File("kv_rearrange/kv_rearrange.metal")
    }

    fn function_name(&self) -> &'static str {
        "kv_rearrange_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        KvRearrangeParams::METAL_STRUCT_DEF.to_string()
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
        todo!("KvRearrange kernel does not yet support compound kernel staging")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_rearrange_params_metal_struct() {
        let def = KvRearrangeParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct KvRearrangeParams"));
        assert!(def.contains("kv_dim"));
        assert!(def.contains("n_heads"));
        assert!(def.contains("n_kv_heads"));
    }
}
