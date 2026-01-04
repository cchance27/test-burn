//! KV Rearrange Kernel for Foundry.
//!
//! Rearranges QKV outputs from [batch*seq, kv_dim] layout to [batch*n_heads, seq, head_dim]
//! for attention computation. Handles GQA (Grouped Query Attention) via group_size.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{foundry::spec::DynamicValue, types::TensorArg};

/// Parameters for KV Rearrange kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvRearrangeParams {
    /// KV dimension (total features in K or V).
    #[serde(default)]
    pub kv_dim: DynamicValue<u32>,
    /// Row stride in input tensor.
    #[serde(default)]
    pub row_stride: DynamicValue<u32>,
    /// Dimension per KV head.
    #[serde(default)]
    pub kv_head_dim: DynamicValue<u32>,
    /// Number of query heads.
    #[serde(default)]
    pub n_heads: DynamicValue<u32>,
    /// Number of KV heads (for GQA).
    #[serde(default)]
    pub n_kv_heads: DynamicValue<u32>,
    /// Dimension per output head.
    #[serde(default)]
    pub head_dim: DynamicValue<u32>,
    /// Sequence length.
    #[serde(default)]
    pub seq: DynamicValue<u32>,
    /// Total output elements.
    #[serde(default)]
    pub total_elements: DynamicValue<u32>,
}

/// KV Rearrange kernel.
///
/// Rearranges [batch*seq, kv_dim] → [batch*n_heads, seq, head_dim].
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_rearrange/kv_rearrange.metal",
    function = "kv_rearrange_kernel_f16",
    args = KvRearrangeParamsResolved,
    dispatch = per_element,
    step = true
)]
pub struct KvRearrange {
    /// Input tensor [batch*seq, kv_dim].
    pub input: TensorArg,
    /// Output tensor [batch*n_heads, seq, head_dim].
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: KvRearrangeParamsResolved,
}

impl KvRearrange {}

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
