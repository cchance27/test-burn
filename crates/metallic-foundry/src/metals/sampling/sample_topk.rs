//! SampleTopKTopP Kernel for Foundry.
//!
//! Fused Top-K + Top-P + Sampling kernel.
//! Single-stage fusion: Min-heap selection -> Sorting -> Softmax -> Cumulative Sum -> Sampling.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize};

/// Parameters for SampleTopK kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct SampleParams {
    pub vocab_size: u32,
    pub k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub seed: u32,
    pub per_thread_m: u32,
    pub num_threadgroups: u32,
}

/// SampleTopKTopP kernel.
///
/// Input: logits [vocab_size]
/// Output: token_id [1]
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "sampling/sample_topk.metal",
    function = "sample_topk_fused_f16",
    args = SampleParams,
    dispatch = true,
    dtype = F16
)]
pub struct SampleTopK {
    /// Input logits [vocab_size].
    pub logits: TensorArg,
    /// Output token index [1].
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: SampleParams,
    /// Threads per threadgroup (must match dispatch_config()).
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl SampleTopK {
    /// Create a new SampleTopK kernel.
    pub fn new(
        logits: &TensorArg,
        output: &TensorArg,
        vocab_size: u32,
        k: u32,
        top_p: f32,
        min_p: f32,
        temperature: f32,
        seed: u32,
    ) -> Self {
        const THREADS_PER_GROUP_DEFAULT: u32 = 1024;
        const PER_THREAD_M_CLAMP_DEFAULT: u32 = 40;

        let threads_per_group = std::env::var("METALLIC_SAMPLE_TPTG")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0 && v <= THREADS_PER_GROUP_DEFAULT as usize)
            .unwrap_or(THREADS_PER_GROUP_DEFAULT as usize);

        // Logic from legacy select_default_per_thread_m
        let default_m = if threads_per_group >= 1024 {
            6
        } else if threads_per_group >= 896 {
            4
        } else if threads_per_group >= 320 {
            6
        } else if threads_per_group >= 256 {
            5
        } else if threads_per_group >= 192 {
            4
        } else {
            3
        };

        let per_thread_m_clamp = PER_THREAD_M_CLAMP_DEFAULT;
        let override_m = std::env::var("METALLIC_SAMPLE_PER_THREAD_M")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&v| v >= 1);
        let per_thread_m = override_m.unwrap_or(default_m).min(k).min(per_thread_m_clamp).max(1);

        Self {
            logits: logits.clone(),
            output: output.clone(),
            params: SampleParams {
                vocab_size,
                k,
                top_p,
                min_p,
                temperature,
                seed,
                per_thread_m,
                num_threadgroups: 1, // Currently single-block logic
            },
            threads_per_threadgroup: threads_per_group,
        }
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            grid: GridSize::d1(1), // Single threadgroup for now
            group: ThreadgroupSize::d1(self.threads_per_threadgroup),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_params_metal_struct() {
        let def = SampleParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct SampleParams"));
        assert!(def.contains("vocab_size"));
        assert!(def.contains("per_thread_m"));
    }
}
