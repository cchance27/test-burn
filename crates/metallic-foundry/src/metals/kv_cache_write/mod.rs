//! KV Cache kernels for Foundry DSL.
//!
//! - `KvCacheWrite`: Writes current K/V into persistent cache at position
//! - `KvCacheRead`: Reads cache slice [0..seq_len] for SDPA consumption

mod read;

use metallic_macros::{Kernel, KernelArgs, MetalStruct};
pub use read::{KvCacheRead, KvCacheReadParams, KvCacheReadParamsResolved};

use crate::{
    spec::DynamicValue, types::TensorArg
};

/// Parameters for KvCacheWriteRepeatKvHeads kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvCacheWriteRepeatKvHeadsParams {
    /// Number of KV heads (input heads).
    pub n_kv_heads: u32,
    /// Number of query heads (output heads).
    pub n_heads: u32,
    /// Group size (n_heads / n_kv_heads).
    pub group_size: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Input sequence length (number of tokens in this step/prefill).
    pub input_seq_len: DynamicValue<u32>,
    /// Position to write to in cache (dynamic).
    pub position_offset: DynamicValue<u32>,
    /// Maximum sequence length (stride dimension in cache).
    pub max_seq_len: DynamicValue<u32>,
    /// Total input elements to copy (n_kv_heads * input_seq_len * head_dim).
    pub total_elements: DynamicValue<u32>,
    /// Layer index for cache selection (unused by kernel, kept for spec compatibility).
    pub layer_idx: DynamicValue<u32>,
}

/// KvCacheWriteRepeatKvHeads kernel.
///
/// Copies the current K/V tensor into the expanded cache at position_offset,
/// repeating KV heads across query heads (GQA).
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_cache_write/kv_cache_write_repeat.metal",
    function = "kv_cache_write_repeat_kv_heads_kernel",
    args = "KvCacheWriteRepeatKvHeadsParamsResolved",
    dtype = "F16",
    dispatch = per_element,
    step = true
)]
pub struct KvCacheWriteRepeatKvHeads {
    /// Input K or V tensor [n_kv_heads, input_seq_len, head_dim].
    pub input: TensorArg,
    /// Output cache [n_heads, max_seq_len, head_dim].
    #[arg(output)]
    pub cache: TensorArg,
    /// Kernel parameters.
    pub params: KvCacheWriteRepeatKvHeadsParamsResolved,
}

/// Parameters for KvCacheWrite kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvCacheWriteParams {
    /// Number of KV heads.
    pub n_kv_heads: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Input sequence length (number of tokens in this step/prefill).
    pub input_seq_len: DynamicValue<u32>,
    /// Position to write to in cache (dynamic).
    pub position_offset: DynamicValue<u32>,
    /// Maximum sequence length (stride dimension in cache).
    pub max_seq_len: DynamicValue<u32>,
    /// Total elements to copy (n_kv_heads * head_dim).
    pub total_elements: DynamicValue<u32>,
    /// Layer index for cache selection.
    pub layer_idx: DynamicValue<u32>,
}

/// KvCacheWrite kernel.
///
/// Copies the current K/V tensor into the cache at position_offset.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_cache_write/kv_cache_write.metal",
    function = "kv_cache_write_kernel",
    args = "KvCacheWriteParamsResolved",
    dtype = "F16",
    dispatch = per_element,
    step = true
)]
pub struct KvCacheWrite {
    /// Input K or V tensor [1, n_kv_heads, head_dim].
    pub input: TensorArg,
    /// Output cache [n_kv_heads, max_seq_len, head_dim].
    #[arg(output)]
    pub cache: TensorArg,
    /// Kernel parameters.
    pub params: KvCacheWriteParamsResolved,
}
