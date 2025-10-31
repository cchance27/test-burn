use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, operation::ComputeKernelEncoder};
mod silu_test;

/// Public, user-facing, zero-sized struct for the SiLU operation.
pub struct SiluOp;

/// Internal struct that holds data for the Operation trait.
struct Silu<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for SiluOp {
    type Args<'a, T: TensorElement> = Tensor<T>;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Silu)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        input: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        ctx.prepare_tensors_for_active_cmd(&[&input])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("silu_op"));

        let op = Silu {
            input,
            output: output.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for Silu<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_1d(self.input.len() as u32, 256);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        set_buffer(encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(encoder, 1, &self.output.buf, self.output.offset);
        set_bytes(encoder, 2, &(self.input.len() as u32));
    }
}
