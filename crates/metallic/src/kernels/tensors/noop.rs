use super::*;
use crate::{
    CommandBuffer, TensorElement, encoder::{dispatch_threadgroups, set_compute_pipeline_state}
};

/// Public, user-facing, zero-sized struct for a NOOP operation.
/// This runs a minimal compute kernel that does nothing and returns the provided tensor unchanged.
pub struct NoopOp;

/// Internal struct implementing the Operation trait.
struct Noop<T: TensorElement> {
    #[allow(dead_code)]
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for NoopOp {
    type Args<'a, T: TensorElement> = Tensor<T>;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Noop)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        out: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Ensure the tensor is tracked for an active command.
        ctx.prepare_tensors_for_active_cmd(&[&out])?;

        let op = Noop {
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for Noop<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        set_compute_pipeline_state(&encoder, &self.pipeline);

        // Minimal threadgroup dispatch: 1 thread, no buffers.
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
        dispatch_threadgroups(&encoder, threadgroups, threadgroup_size);

        Ok(())
    }
}
