use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, operation::ComputeKernelEncoder};

pub struct ElemwiseDivOp;

#[cfg(test)]
mod elemwise_div_test;

struct ElemwiseDiv<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for ElemwiseDivOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseDiv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        if a.dims() != b.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseDiv: input shapes must match, got a={:?}, b={:?}",
                a.dims(),
                b.dims(),
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("elemwise_div_op"));

        let op = ElemwiseDiv {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for ElemwiseDiv<T> {
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
