use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use super::*;
use crate::{CommandBuffer, TensorElement, context::GpuProfilerLabel, operation::ComputeKernelEncoder};

/// Public, user-facing, zero-sized struct for a NOOP operation.
/// This runs a minimal compute kernel that does nothing and returns the provided tensor unchanged.
pub struct NoopOp;

/// Internal struct implementing the Operation trait.
struct Noop<T: TensorElement> {
    #[allow(dead_code)]
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for NoopOp {
    type Args<'a, T: TensorElement> = Tensor<T>;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Noop)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        out: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Ensure the tensor is tracked for an active command.
        ctx.prepare_tensors_for_active_cmd(&[&out])?;

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("noop_op"));

        let op = Noop {
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for Noop<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(threadgroups, threadgroup_size);

        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        // No arguments to bind for a NOOP operation
    }
}
