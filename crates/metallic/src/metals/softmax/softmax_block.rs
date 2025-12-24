//! SoftmaxBlock Kernel for Foundry.
//!
//! Block-based softmax using segmented reductions for very long sequences.
//! Each threadgroup handles a segment, with cross-segment reduction.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{
    tensor::Dtype, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for SoftmaxBlock kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct SoftmaxBlockParams {
    pub batch: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub segment_size: u32,
    pub causal: u32,
    pub query_offset: u32,
}

/// SoftmaxBlock kernel.
///
/// Input: attention scores [batch, seq_q, seq_k]
/// Output: attention weights [batch, seq_q, seq_k]
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "softmax/softmax_block.metal",
    function = "softmax_block_f16",
    stage_function = "run_softmax_block_core",
    args = "SoftmaxBlockParams",
    threadgroup = "float shared_max[256]; float shared_sum[256]"
)]
pub struct SoftmaxBlock {
    /// Input attention scores (Buffer 0 - Policy Matrix).
    #[arg(buffer = 0, stage_skip)]
    pub input: TensorArg,
    /// Scale bytes for Q8 policy (Buffer 1 - Policy Scales).
    #[arg(buffer = 1, stage_skip)]
    pub scale_bytes: TensorArg,
    /// Output attention weights (Buffer 2).
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    /// Batch size.
    #[arg(buffer = 3)]
    pub batch: u32,
    /// seq_q parameter.
    #[arg(buffer = 4)]
    pub seq_q: u32,
    /// seq_k parameter.
    #[arg(buffer = 5)]
    pub seq_k: u32,
    /// Segment size for blocked processing.
    #[arg(buffer = 6)]
    pub segment_size: u32,
    /// Causal mask flag.
    #[arg(buffer = 7)]
    pub causal: u32,
    /// Query offset for causal masking.
    #[arg(buffer = 8)]
    pub query_offset: u32,
}

impl SoftmaxBlock {
    /// Create a new SoftmaxBlock kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, batch: u32, seq_q: u32, seq_k: u32, causal: bool, query_offset: u32) -> Self {
        const SEGMENT_SIZE: u32 = 1024; // Fixed segment size
        Self {
            input: input.clone(),
            output: output.clone(),
            scale_bytes: input.clone(), // Default: duplicate input for scales
            batch,
            seq_q,
            seq_k,
            segment_size: SEGMENT_SIZE,
            causal: causal as u32,
            query_offset,
        }
    }

    /// Dispatch configuration - required by Kernel derive.
    pub fn dispatch_config(&self) -> DispatchConfig {
        const THREADS_PER_TG: usize = 256;
        let rows = (self.batch * self.seq_q) as usize;

        DispatchConfig {
            grid: GridSize::d1(rows),
            group: ThreadgroupSize::d1(THREADS_PER_TG),
        }
    }

    /// dtype
    #[allow(dead_code)]
    pub fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }
}
