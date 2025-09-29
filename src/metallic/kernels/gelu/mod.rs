use super::*;
use crate::metallic::{TensorInit, TensorStorage};

pub struct GeluOp;

struct Gelu {
    input: Tensor,
    output: Tensor,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for GeluOp {
    type Args<'a> = Tensor;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Gelu)
    }

    fn new<'a>(
        ctx: &mut Context,
        input: Self::Args<'a>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let input = input;
        ctx.prepare_tensors_for_active_cmd(&[&input])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let op = Gelu {
            input,
            output: output.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for Gelu {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.input.len() as u32;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_bytes(&encoder, 2, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod gelu_test;
