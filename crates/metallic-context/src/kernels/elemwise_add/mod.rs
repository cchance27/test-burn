use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, operation::ComputeKernelEncoder};

// Additional Operations for this Metal Kernel (additional functions in the kernel)
mod elemwise_broadcast_add;
#[cfg(test)]
mod elemwise_broadcast_add_test;
pub use elemwise_broadcast_add::{BroadcastElemwiseAddInplaceOp, BroadcastElemwiseAddOp};

// Tests for Main Operation
#[cfg(test)]
mod elemwise_add_test;

// User-facing struct for the standard element-wise add operation.
pub struct ElemwiseAddOp;

// Internal struct that holds the operation data.
struct ElemwiseAdd<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for ElemwiseAddOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseAdd)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        if a.dims() != b.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseAdd: input shapes must match, got a={:?}, b={:?}",
                a.dims(),
                b.dims(),
            )));
        }
        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;
        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("elemwise_add_op"));
        let op = ElemwiseAdd {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };
        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for ElemwiseAdd<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_1d(self.a.len() as u32, 256);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        set_buffer(encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(encoder, 3, &(self.a.len() as u32));
    }
}
