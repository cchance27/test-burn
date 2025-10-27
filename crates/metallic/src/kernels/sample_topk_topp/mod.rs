use std::sync::atomic::{AtomicU32, Ordering};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use super::*;
use crate::{CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, resource_cache::ResourceCache, tensor::dtypes::U32};

// Global counter for seed variation
static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);

pub fn calculate_threads_per_tg_and_num_threadgroups(
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    vocab_size: u32,
) -> (usize, u32) {
    let tew = pipeline.threadExecutionWidth() as usize;
    let max_tptg = pipeline.maxTotalThreadsPerThreadgroup() as usize;

    // Choose threads_per_tg as a multiple of TEW, capped at 256 and device max
    let max_cap = 256usize.min(max_tptg);
    let multiples = (max_cap / tew).max(1);
    let threads_per_tg = (multiples * tew).min(max_cap);

    // Increase threadgroups for large vocabs to improve coverage and occupancy
    let items_per_thread = 32u32; // moderate per-thread work to reduce divergence
    let items_per_tg = (threads_per_tg as u32) * items_per_thread;
    let num_tgs = vocab_size.div_ceil(items_per_tg).clamp(1, 128);

    (threads_per_tg, num_tgs as u32)
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

pub struct SampleTopKTopPOp;

impl CustomKernelInvocable for SampleTopKTopPOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, f32, f32, u32, u32);
    type OutputTuple<T: TensorElement> = (U32,);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, <Self::OutputTuple<T> as MultiTensorOutput<T>>::Tensors), MetalError> {
        let (logits, vocab_size, k, top_p, temperature, base_seed, per_thread_m_clamp) = args;

        // CRITICAL: Vary the seed on each call!
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let varied_seed = base_seed.wrapping_add(counter);

        let (partials_values, partials_indices) =
            ctx.call_custom::<SampleTopKPartialsOp>((logits.clone(), vocab_size, k, top_p, temperature, varied_seed, per_thread_m_clamp))?;

        let params = SampleParams {
            vocab_size,
            k,
            top_p,
            temperature,
            seed: varied_seed, // Use the varied seed
            per_thread_m: k.clamp(1, per_thread_m_clamp),
            num_threadgroups: 0, // will be set in partials/merge ops consistently
        };

        let (output_token,) = ctx.call_custom::<SampleTopKMergeAndSampleOp>((partials_values, partials_indices, params))?;

        let op = Box::new(SampleTopKWrapper);
        Ok((op, (output_token,)))
    }
}

struct SampleTopKWrapper;
impl Operation for SampleTopKWrapper {
    fn encode(&self, _cb: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        Ok(())
    }
}
