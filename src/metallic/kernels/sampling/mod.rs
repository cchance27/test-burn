use super::*;

use crate::metallic::instrumentation::{KernelDispatchKind, SamplingDispatchTiming};
use crate::metallic::sampling::effective_top_k;
use crate::metallic::{TensorElement, tensor::RetainedBuffer};
use objc2::msg_send;
use objc2::runtime::{Bool, ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCounterSampleBuffer, MTLResourceOptions};
use std::mem;

/// Number of threads launched for the sampling reduction kernel. This must match
/// `THREADGROUP_SIZE` in `kernel.metal`.
const THREADGROUP_SIZE: usize = 8;

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
    completion_counter: RetainedBuffer,
    params: SamplingParams,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dispatch_timing: Option<SamplingDispatchTiming>,
    effective_top_k: usize,
    threadgroup_count: usize,
}

impl KernelInvocable for SampleTopKTopPOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, SamplingParams, RetainedBuffer);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SampleTopKTopP)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F32 {
            return Err(MetalError::OperationNotSupported(
                "top-k/top-p sampling kernel only supports f32 logits".to_string(),
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
        let threadgroup_count = ((vocab_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE).max(1);
        if threadgroup_count > u32::MAX as usize {
            return Err(MetalError::OperationNotSupported(
                "sampling kernel threadgroup count exceeds u32::MAX".to_string(),
            ));
        }

        let effective_top_k = effective_top_k(raw_params.top_k as usize, vocab_size).min(MAX_TOP_K);

        let partial_len = threadgroup_count
            .checked_mul(effective_top_k)
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel partial buffer size overflow".to_string()))?;

        let partial_vals_bytes = partial_len
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel partial values byte size overflow".to_string()))?;
        let partial_indices_bytes = partial_len
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel partial indices byte size overflow".to_string()))?;

        let counts_bytes = threadgroup_count
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel counts byte size overflow".to_string()))?;

        let fallback_bytes = threadgroup_count
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel fallback values byte size overflow".to_string()))?;

        let fallback_index_bytes = threadgroup_count
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| MetalError::OperationNotSupported("sampling kernel fallback indices byte size overflow".to_string()))?;

        let partial_vals = ctx
            .device
            .newBufferWithLength_options(partial_vals_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(partial_vals_bytes))?;
        let partial_indices = ctx
            .device
            .newBufferWithLength_options(partial_indices_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(partial_indices_bytes))?;
        let partial_counts = ctx
            .device
            .newBufferWithLength_options(counts_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(counts_bytes))?;
        let fallback_vals = ctx
            .device
            .newBufferWithLength_options(fallback_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(fallback_bytes))?;
        let fallback_indices = ctx
            .device
            .newBufferWithLength_options(fallback_index_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(fallback_index_bytes))?;
        let fallback_flags = ctx
            .device
            .newBufferWithLength_options(fallback_index_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(fallback_index_bytes))?;

        let completion_counter_bytes = mem::size_of::<u32>();
        let completion_counter = ctx
            .device
            .newBufferWithLength_options(completion_counter_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(completion_counter_bytes))?;

        unsafe {
            let ptr = completion_counter.contents().as_ptr().cast::<u32>();
            ptr.write_unaligned(0);
        }

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
            completion_counter,
            params,
            pipeline: pipeline.expect("Kernel Module should supply a pipeline"),
            dispatch_timing,
            effective_top_k,
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

        unsafe {
            let ptr = self.completion_counter.contents().as_ptr().cast::<u32>();
            ptr.write_unaligned(0);
        }

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

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.logits.buf, self.logits.offset);
        set_buffer(&encoder, 1, &self.result, 0);
        set_buffer(&encoder, 2, &self.partial_vals, 0);
        set_buffer(&encoder, 3, &self.partial_indices, 0);
        set_buffer(&encoder, 4, &self.partial_counts, 0);
        set_buffer(&encoder, 5, &self.fallback_vals, 0);
        set_buffer(&encoder, 6, &self.fallback_indices, 0);
        set_buffer(&encoder, 7, &self.fallback_flags, 0);
        set_buffer(&encoder, 8, &self.completion_counter, 0);
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
