use std::sync::atomic::{AtomicU32, Ordering};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState};

use super::*;
use crate::{CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache, tensor::dtypes::U32};

// Global counter for seed variation
static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);

mod sample_topk_fused;
use sample_topk_fused::SampleTopKFusedOp;

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

        let (output_token,) =
            ctx.call_custom::<SampleTopKFusedOp>((logits.clone(), vocab_size, k, top_p, temperature, varied_seed, per_thread_m_clamp))?;
        let op = Box::new(SampleTopKWrapper);
        Ok((op, (output_token,)))
    }
}

struct SampleTopKWrapper;
impl Operation for SampleTopKWrapper {
    fn encode(&self, _cb: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        // No arguments to bind for this placeholder operation
    }
}
