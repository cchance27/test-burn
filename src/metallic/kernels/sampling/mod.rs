use super::*;
use crate::metallic::{Dtype, MetalError, Tensor, TensorElement, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLComputePipelineState, MTLSize};
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SamplerConfig {
    pub vocab_size: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub rng_seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SamplerResult {
    pub selected: u32,
    pub fallback: u32,
    pub used_fallback: u32,
    pub padding: u32,
}

pub struct SamplerOperation<T: TensorElement> {
    pub logits: Tensor<T>,
    pub shortlist_values: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub shortlist_indices: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub result: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub config: SamplerConfig,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl<T: TensorElement> Operation for SamplerOperation<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.logits.buf, self.logits.offset);
        set_buffer(&encoder, 1, &self.shortlist_values, 0);
        set_buffer(&encoder, 2, &self.shortlist_indices, 0);
        set_buffer(&encoder, 3, &self.result, 0);
        set_bytes(&encoder, 4, &self.config);

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();
        Ok(())
    }
}

impl SamplerConfig {
    pub fn size() -> usize {
        size_of::<SamplerConfig>()
    }
}

impl SamplerResult {
    pub fn size() -> usize {
        size_of::<SamplerResult>()
    }
}

pub fn kernel_function_for_dtype(dtype: Dtype) -> Result<KernelFunction, MetalError> {
    match dtype {
        Dtype::F32 => Ok(KernelFunction::SamplerF32),
        Dtype::F16 => Ok(KernelFunction::SamplerF16),
        other => Err(MetalError::UnsupportedDtype {
            operation: "gpu_sampler",
            dtype: other,
        }),
    }
}
