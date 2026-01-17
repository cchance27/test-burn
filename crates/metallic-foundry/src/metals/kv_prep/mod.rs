//! Fused KV preparation kernel(s).
//!
//! This fuses the common decode-time KV preparation chain:
//! - KvRearrange(Q) + Rope(Q) -> q_rot
//! - KvRearrange(K) + Rope(K) + KvCacheWriteRepeatKvHeads(K) -> k_cache
//! - KvRearrange(V) + KvCacheWriteRepeatKvHeads(V) -> v_cache
//!
//! The target is to reduce per-token dispatch fanout during decode.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{spec::DynamicValue, types::TensorArg};

#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvPrepFusedParams {
    /// d_model (n_heads * head_dim).
    pub d_model: u32,
    /// kv_dim (n_kv_heads * head_dim).
    pub kv_dim: u32,
    /// head_dim (must be even for RoPE).
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// n_heads / n_kv_heads.
    pub group_size: u32,
    /// Number of tokens processed in this step.
    pub seq_len: DynamicValue<u32>,
    /// Offset into the KV cache / RoPE tables.
    pub position_offset: DynamicValue<u32>,
    /// KV cache stride dimension.
    pub max_seq_len: DynamicValue<u32>,
    /// Total Q elements (n_heads * seq_len * head_dim).
    /// Per-element dispatch expects this name.
    pub total_elements: DynamicValue<u32>,
}

/// Fused KV-prep kernel for F16 activations.
///
/// Constraints:
/// - head_dim must be even (RoPE pairs)
/// - n_heads must be divisible by n_kv_heads (GQA group_size)
/// - Inputs are row-major with row_stride == dim
/// - Assumes batch == 1 (matches current decode/prefill usage)
/// - position_offset + seq_len must be <= max_seq_len
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_prep/kv_prep_fused.metal",
    function = "kv_prep_fused_kernel_f16",
    args = KvPrepFusedParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = true
)]
pub struct KvPrepFused {
    pub q: TensorArg,
    pub k: TensorArg,
    pub v: TensorArg,

    #[arg(output)]
    pub q_rot: TensorArg,
    #[arg(output)]
    pub k_cache: TensorArg,
    #[arg(output)]
    pub v_cache: TensorArg,

    pub cos: TensorArg,
    pub sin: TensorArg,

    pub params: KvPrepFusedParamsResolved,
}
