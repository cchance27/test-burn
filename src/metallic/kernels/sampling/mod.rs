use super::*;

use crate::metallic::{TensorElement, tensor::RetainedBuffer};

/// Maximum supported top-k for the GPU sampling kernel. Larger requests fall
/// back to the CPU implementation to avoid excessive per-thread stack usage.
pub const MAX_TOP_K: usize = 256;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingParams {
    pub vocab_size: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub random_u32: u32,
    pub _padding: u32,
}

pub struct SampleTopKTopPOp;

struct SampleTopKTopP<T: TensorElement> {
    logits: Tensor<T>,
    result: RetainedBuffer,
    params: SamplingParams,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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

        let (logits, params, result) = args;
        ctx.prepare_tensors_for_active_cmd(&[&logits])?;

        let output = logits.clone();
        let op = SampleTopKTopP {
            logits,
            result,
            params,
            pipeline: pipeline.expect("Kernel Module should supply a pipeline"),
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

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.logits.buf, self.logits.offset);
        set_buffer(&encoder, 1, &self.result, 0);
        set_bytes(&encoder, 2, &self.params);

        let grid_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();
        Ok(())
    }
}
