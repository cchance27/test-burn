use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize};

#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct RepetitionPenaltyParams {
    pub vocab_size: u32,
    pub recent_len: u32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// Apply repetition/presence/frequency penalties in-place to logits for a list of recent token IDs.
///
/// This kernel uses a packed `(token_id, count)` input so repeated tokens apply `penalty^count`
/// deterministically without a write race on `logits[token_id]`.
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "sampling/repetition_penalty.metal",
    function = "apply_repetition_penalty_f16",
    args = RepetitionPenaltyParams,
    dispatch = true,
    dtype = F16
)]
pub struct ApplyRepetitionPenalty {
    /// Logits [vocab_size] (in-place).
    #[arg(output)]
    pub logits: TensorArg,
    /// Packed pairs [recent_len * 2]: (token_id, count).
    #[arg(metal_type = "const device uint*")]
    pub recent_pairs: TensorArg,
    pub params: RepetitionPenaltyParams,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl ApplyRepetitionPenalty {
    pub fn new(
        logits: &TensorArg,
        recent_pairs: &TensorArg,
        vocab_size: u32,
        pair_len: u32,
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> Self {
        let threads_per_threadgroup = 256usize;
        Self {
            logits: logits.clone(),
            recent_pairs: recent_pairs.clone(),
            params: RepetitionPenaltyParams {
                vocab_size,
                recent_len: pair_len,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
            },
            threads_per_threadgroup,
        }
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        let n = self.params.recent_len.max(1) as usize;
        DispatchConfig {
            grid: GridSize::d1(n),
            group: ThreadgroupSize::d1(self.threads_per_threadgroup),
        }
    }
}
