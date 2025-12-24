//! SampleTopKTopP Kernel for Foundry.
//!
//! Fused Top-K + Top-P + Sampling kernel.
//! Single-stage fusion: Min-heap selection -> Sorting -> Softmax -> Cumulative Sum -> Sampling.

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for SampleTopK kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct SampleParams {
    pub vocab_size: u32,
    pub k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: u32,
    pub per_thread_m: u32,
    pub num_threadgroups: u32,
}

/// SampleTopKTopP kernel.
///
/// Input: logits [vocab_size]
/// Output: token_id [1]
#[derive(KernelArgs, Clone)]
pub struct SampleTopK {
    /// Input logits [vocab_size].
    #[arg(buffer = 0)]
    pub logits: TensorArg,
    /// Output token index [1].
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 2)]
    pub params: SampleParams,
}

impl SampleTopK {
    /// Create a new SampleTopK kernel.
    pub fn new(logits: &TensorArg, output: &TensorArg, vocab_size: u32, k: u32, top_p: f32, temperature: f32, seed: u32) -> Self {
        const THREADS_PER_GROUP: u32 = 1024;

        // Logic from legacy select_default_per_thread_m
        let default_m = if THREADS_PER_GROUP >= 1024 {
            6
        } else if THREADS_PER_GROUP >= 896 {
            4
        } else if THREADS_PER_GROUP >= 320 {
            6
        } else if THREADS_PER_GROUP >= 256 {
            5
        } else {
            3
        };

        // Clamp logic
        let per_thread_m_clamp = 6; // Typical default max
        let per_thread_m = default_m.min(k).min(per_thread_m_clamp).max(1);

        Self {
            logits: logits.clone(),
            output: output.clone(),
            params: SampleParams {
                vocab_size,
                k,
                top_p,
                temperature,
                seed,
                per_thread_m,
                num_threadgroups: 1, // Currently single-block logic
            },
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct SampleTopKId;

impl Kernel for SampleTopK {
    type Args = SampleParams;
    type Id = SampleTopKId;

    fn source(&self) -> KernelSource {
        KernelSource::File("sampling/sample_topk.metal")
    }

    fn function_name(&self) -> &'static str {
        "sample_topk_fused_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        SampleParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            grid: GridSize::d1(1), // Single threadgroup for now
            group: ThreadgroupSize::d1(1024),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("SampleTopK kernel does not yet support compound kernel staging")
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
