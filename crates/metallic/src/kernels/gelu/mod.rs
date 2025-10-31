use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage};

pub struct GeluOp;

struct Gelu<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl DefaultKernelInvocable for GeluOp {
    type Args<'a, T: TensorElement> = Tensor<T>;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Gelu)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        input: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
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

impl<T: TensorElement> Operation for Gelu<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

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
        Ok(())
    }
}

#[cfg(test)]
mod gelu_test;
