use super::*;

use crate::metallic::instrumentation::{KernelDispatchKind, SamplingDispatchTiming};
use crate::metallic::sampling::effective_top_k;
use crate::metallic::{TensorElement, tensor::RetainedBuffer};
use objc2::msg_send;
use objc2::runtime::{Bool, ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::MTLCounterSampleBuffer;

/// Number of threads launched for the sampling reduction kernel. This must match
/// `THREADGROUP_SIZE` in `kernel.metal`.
const THREADGROUP_SIZE: usize = 64;

/// Number of vocabulary elements processed by each thread within a threadgroup.
const TOKENS_PER_THREAD: usize = 16;

/// Total number of vocabulary elements covered by a single threadgroup.
const THREADGROUP_TOKENS: usize = THREADGROUP_SIZE * TOKENS_PER_THREAD;

/// Maximum supported top-k for the GPU sampling kernel. Larger requests fall
/// back to the CPU implementation to avoid excessive per-thread stack usage.
pub const MAX_TOP_K: usize = 256;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingParams {
    pub vocab_size: u32,
    pub top_k: u32,
    pub random_u32: u32,
    pub threadgroup_count: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub _padding0: u32,
    pub _padding1: u32,
}

pub struct SampleTopKTopPOp;

struct SampleTopKTopP<T: TensorElement> {
    logits: Tensor<T>,
    result: RetainedBuffer,
    partial_vals: RetainedBuffer,
    partial_indices: RetainedBuffer,
    partial_counts: RetainedBuffer,
    fallback_vals: RetainedBuffer,
    fallback_indices: RetainedBuffer,
    fallback_flags: RetainedBuffer,
    params: SamplingParams,
    stage1_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    stage2_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dispatch_timing: Option<SamplingDispatchTiming>,
    threadgroup_count: usize,
}

impl KernelInvocable for SampleTopKTopPOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, SamplingParams, RetainedBuffer);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SampleTopKTopPStage1)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if !matches!(T::DTYPE, Dtype::F32 | Dtype::F16) {
            return Err(MetalError::OperationNotSupported(
                "top-k/top-p sampling kernel only supports f32 or f16 logits".to_string(),
            ));
        }

        let (logits, raw_params, result) = args;
        ctx.prepare_tensors_for_active_cmd(&[&logits])?;

        let dispatch_timing = {
            let command_buffer = {
                let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
                command_buffer.clone()
            };
            ctx.register_sampling_dispatch(&command_buffer).timing().cloned()
        };

        let vocab_size = raw_params.vocab_size as usize;
        let threadgroup_count = ((vocab_size + THREADGROUP_TOKENS - 1) / THREADGROUP_TOKENS).max(1);
        if threadgroup_count > u32::MAX as usize {
            return Err(MetalError::OperationNotSupported(
                "sampling kernel threadgroup count exceeds u32::MAX".to_string(),
            ));
        }

        let effective_top_k = effective_top_k(raw_params.top_k as usize, vocab_size).min(MAX_TOP_K);

        let scratch = ctx.acquire_sampling_scratch(threadgroup_count, effective_top_k)?;
        let partial_vals = scratch.partial_vals;
        let partial_indices = scratch.partial_indices;
        let partial_counts = scratch.partial_counts;
        let fallback_vals = scratch.fallback_vals;
        let fallback_indices = scratch.fallback_indices;
        let fallback_flags = scratch.fallback_flags;

        let stage1_pipeline = pipeline.expect("Kernel Module should supply a pipeline");
        let stage2_pipeline = ctx
            .kernel_manager
            .get_pipeline(KernelFunction::SampleTopKTopPFinalize, T::DTYPE, &ctx.device)?;

        let params = SamplingParams {
            vocab_size: raw_params.vocab_size,
            top_k: effective_top_k as u32,
            random_u32: raw_params.random_u32,
            threadgroup_count: threadgroup_count as u32,
            top_p: raw_params.top_p,
            temperature: raw_params.temperature,
            _padding0: 0,
            _padding1: 0,
        };

        let output = logits.clone();
        let op = SampleTopKTopP {
            logits,
            result,
            partial_vals,
            partial_indices,
            partial_counts,
            fallback_vals,
            fallback_indices,
            fallback_flags,
            params,
            stage1_pipeline,
            stage2_pipeline,
            dispatch_timing,
            threadgroup_count,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for SampleTopKTopP<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        if let Some(timing) = &self.dispatch_timing
            && matches!(timing.kind(), KernelDispatchKind::Compute)
        {
            let sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer> = timing.sample_buffer();
            unsafe {
                let _: () = msg_send![
                    &*encoder,
                    sampleCountersInBuffer: sample_buffer,
                    atSampleIndex: timing.start_index(),
                    withBarrier: Bool::YES
                ];
            }
        }

        set_compute_pipeline_state(&encoder, &self.stage1_pipeline);
        set_buffer(&encoder, 0, &self.logits.buf, self.logits.offset);
        set_buffer(&encoder, 1, &self.result, 0);
        set_buffer(&encoder, 2, &self.partial_vals, 0);
        set_buffer(&encoder, 3, &self.partial_indices, 0);
        set_buffer(&encoder, 4, &self.partial_counts, 0);
        set_buffer(&encoder, 5, &self.fallback_vals, 0);
        set_buffer(&encoder, 6, &self.fallback_indices, 0);
        set_buffer(&encoder, 7, &self.fallback_flags, 0);
        set_bytes(&encoder, 9, &self.params);

        let threadgroup_size = MTLSize {
            width: THREADGROUP_SIZE as NSUInteger,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: self.threadgroup_count as NSUInteger,
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(&encoder, threadgroups, threadgroup_size);

        encoder.endEncoding();

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        set_compute_pipeline_state(&encoder, &self.stage2_pipeline);
        set_buffer(&encoder, 0, &self.logits.buf, self.logits.offset);
        set_buffer(&encoder, 1, &self.result, 0);
        set_buffer(&encoder, 2, &self.partial_vals, 0);
        set_buffer(&encoder, 3, &self.partial_indices, 0);
        set_buffer(&encoder, 4, &self.partial_counts, 0);
        set_buffer(&encoder, 5, &self.fallback_vals, 0);
        set_buffer(&encoder, 6, &self.fallback_indices, 0);
        set_buffer(&encoder, 7, &self.fallback_flags, 0);
        set_bytes(&encoder, 9, &self.params);

        let finalize_threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let finalize_threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(&encoder, finalize_threadgroups, finalize_threadgroup_size);

        if let Some(timing) = &self.dispatch_timing
            && matches!(timing.kind(), KernelDispatchKind::Compute)
        {
            let sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer> = timing.sample_buffer();
            unsafe {
                let _: () = msg_send![
                    &*encoder,
                    sampleCountersInBuffer: sample_buffer,
                    atSampleIndex: timing.end_index(),
                    withBarrier: Bool::NO
                ];
            }
        }

        encoder.endEncoding();
        Ok(())
    }
}
