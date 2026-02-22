//! RepeatKvHeads Kernel for Foundry.
//!
//! Repeats K/V heads for Grouped Query Attention (GQA).
//! Input: [batch * n_kv_heads, cache_stride, head_dim]
//! Output: [batch * n_heads, seq, head_dim]

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{
    spec::DynamicValue, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for RepeatKvHeads kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct RepeatKvHeadsParams {
    /// Group size (n_heads / n_kv_heads).
    #[serde(default)]
    pub group_size: u32,
    /// Batch size.
    #[serde(default)]
    pub batch: u32,
    /// Number of KV heads.
    #[serde(default)]
    pub n_kv_heads: u32,
    /// Number of query heads.
    #[serde(default)]
    pub n_heads: u32,
    /// Sequence length.
    #[serde(default)]
    pub seq: DynamicValue<u32>,
    /// Head dimension.
    #[serde(default)]
    pub head_dim: u32,
    /// Cache stride (max sequence capacity).
    #[serde(default)]
    pub cache_stride: DynamicValue<u32>,
    /// Total output elements.
    #[serde(default)]
    pub total_elements: DynamicValue<u32>,
}

/// RepeatKvHeads kernel.
///
/// Repeats K/V from n_kv_heads → n_heads for attention.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "repeat_kv_heads/repeat_kv_heads.metal",
    function = "repeat_kv_heads_kernel_f16",
    args = "RepeatKvHeadsParamsResolved",
    step = true
)]
pub struct RepeatKvHeads {
    /// Input tensor [batch * n_kv_heads, cache_stride, head_dim].
    pub input: TensorArg,
    /// Output tensor [batch * n_heads, seq, head_dim].
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: RepeatKvHeadsParamsResolved,
}

impl RepeatKvHeads {
    /// Create a new RepeatKvHeads kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, params: RepeatKvHeadsParamsResolved) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            params,
        }
    }

    /// Dispatch configuration - required by `#[derive(Kernel)]`.
    /// One thread per output element.
    pub fn dispatch_config(&self) -> DispatchConfig {
        let total = self.params.total_elements as usize;
        let threads_per_group = 256;
        let num_groups = total.div_ceil(threads_per_group);

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }
}

#[path = "mod.test.rs"]
mod tests;
