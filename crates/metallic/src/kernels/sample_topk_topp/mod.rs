use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use super::*;
use crate::{
    resource_cache::ResourceCache, tensor::dtypes::U32, CommandBuffer, Context, F32Element, MetalError, Operation, Tensor, TensorElement
};

// Helper function to calculate the number of threadgroups for the partials kernel
// This ensures both the partials and merge kernels use the same calculation
// Uses default values that should match what the pipeline would have
pub fn calculate_threads_per_tg_and_num_threadgroups(pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>, vocab_size: u32) -> (usize, u32) {
    let tew = pipeline.threadExecutionWidth();
    let max_tptg = pipeline.maxTotalThreadsPerThreadgroup();
    let threads_per_tg = tew.max(256).min(max_tptg);    
    (threads_per_tg, vocab_size.div_ceil(threads_per_tg as u32))
}

mod sample_topk_merge_and_sample;
mod sample_topk_partials;
use sample_topk_merge_and_sample::SampleTopKMergeAndSampleOp;
use sample_topk_partials::SampleTopKPartialsOp;

#[cfg(test)]
mod benchmark_test;
#[cfg(test)]
mod top_k_selection_test;
#[cfg(test)]
mod validation_test;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SampleParams {
    pub vocab_size: u32,
    pub k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: u32,
    pub per_thread_m: u32,
    pub num_threadgroups: u32,
}

/// User-facing convenience op that chains the two kernels.
pub struct SampleTopKTopPOp;

impl CustomKernelInvocable for SampleTopKTopPOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, f32, f32, u32, u32);
    type OutputTensor<U: TensorElement> = U32;

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement, U: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<Self::OutputTensor<U>>), MetalError> {
        let (logits, vocab_size, k, top_p, temperature, seed, per_thread_m_clamp) = args;

        // Calculate the expected number of threadgroups that the partials kernel will use        
        let partials = ctx.call_custom::<SampleTopKPartialsOp, F32Element>((logits.clone(), vocab_size, k, top_p, temperature, seed, per_thread_m_clamp))?;               
        let params = SampleParams {
            vocab_size,
            k,
            top_p,
            temperature,
            seed,
            per_thread_m: k.clamp(1, per_thread_m_clamp), // we had this set to 16 previously but it was slower
            num_threadgroups: 0, // will be calculated in the kernel
        };
        let output_token = ctx.call_custom::<SampleTopKMergeAndSampleOp, U32>((partials, params))?;

        // Return a no-op wrapper op (just to satisfy the trait)
        let op = Box::new(SampleTopKWrapper);
        Ok((op, output_token))
    }
}

struct SampleTopKWrapper;
impl Operation for SampleTopKWrapper {
    fn encode(&self, _cb: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        Ok(())
    }
}
