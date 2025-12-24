//! SoftmaxVec Kernel for Foundry.
//!
//! Vectorized softmax using simdgroup reductions for short-to-medium sequences.
//! Each threadgroup processes one row with parallel max/sum reductions.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{
    tensor::Dtype, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for SoftmaxVec kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct SoftmaxVecParams {
    pub seq_q: u32,
    pub seq_k: u32,
    pub causal: u32,
    pub query_offset: u32,
}

/// SoftmaxVec kernel.
///
/// Input: attention scores [batch * seq_q, seq_k]
/// Output: attention weights [batch * seq_q, seq_k]
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "softmax/softmax_vec.metal",
    function = "softmax_vec_f16",
    stage_function = "run_softmax_vec_core",
    args = "SoftmaxVecParams",
    threadgroup = "float shared_data[256]; threadgroup uint shared_indices[256]"
)]
pub struct SoftmaxVec {
    /// Input attention scores (Buffer 0 - Policy Matrix).
    #[arg(buffer = 0, stage_skip)]
    pub input: TensorArg,
    /// Scale bytes for Q8 policy (Buffer 1 - Policy Scales).
    #[arg(buffer = 1, stage_skip)]
    pub scale_bytes: TensorArg,
    /// Output attention weights (Buffer 2).
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    /// seq_q parameter.
    #[arg(buffer = 3)]
    pub seq_q: u32,
    /// seq_k parameter.
    #[arg(buffer = 4)]
    pub seq_k: u32,
    /// Causal mask flag.
    #[arg(buffer = 5)]
    pub causal: u32,
    /// Query offset for causal masking.
    #[arg(buffer = 6)]
    pub query_offset: u32,
    /// Rows total for dispatch.
    rows_total: u32,
}

impl SoftmaxVec {
    /// Create a new SoftmaxVec kernel.
    pub fn new(input: &TensorArg, output: &TensorArg, rows_total: u32, seq_q: u32, seq_k: u32, causal: bool, query_offset: u32) -> Self {
        Self {
            input: input.clone(),
            scale_bytes: input.clone(), // Default: dummy scales (input)
            output: output.clone(),
            seq_q,
            seq_k,
            causal: causal as u32,
            query_offset,
            rows_total,
        }
    }

    /// Dispatch configuration - required by Kernel derive.
    pub fn dispatch_config(&self) -> DispatchConfig {
        // Match legacy: 1 threadgroup per row, threads along X
        let native_width = 32; // Simdgroup width
        DispatchConfig {
            grid: GridSize::d2(1, self.rows_total as usize),
            group: ThreadgroupSize::d1(native_width),
        }
    }

    /// dtype - used by manual Kernel methods still needed
    #[allow(dead_code)]
    pub fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }
}
