//! KvCacheRead Kernel for Foundry DSL.
//!
//! Reads a slice of the KV cache from position 0 to current_seq_len.
//! Output is a contiguous tensor for SDPA consumption.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{
    spec::DynamicValue, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for KvCacheRead kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct KvCacheReadParams {
    /// Number of KV heads.
    pub n_kv_heads: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Current sequence length (position_offset + 1).
    pub seq_len: DynamicValue<u32>,
    /// Maximum sequence length (stride in cache).
    pub max_seq_len: DynamicValue<u32>,
    /// Total elements to read (n_kv_heads * seq_len * head_dim).
    pub total_elements: DynamicValue<u32>,
}

/// KvCacheRead kernel.
///
/// Reads cache[0..seq_len] into a contiguous output tensor.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "kv_cache_write/kv_cache_read.metal",
    function = "kv_cache_read_kernel",
    args = "KvCacheReadParamsResolved",
    dtype = "F16",
    step = true
)]
pub struct KvCacheRead {
    /// Input cache [n_kv_heads, max_seq_len, head_dim].
    pub cache: TensorArg,
    /// Output tensor [n_kv_heads, seq_len, head_dim].
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: KvCacheReadParamsResolved,
}

impl KvCacheRead {
    /// Dispatch configuration.
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
