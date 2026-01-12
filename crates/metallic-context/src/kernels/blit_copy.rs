use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBlitCommandEncoder as _;

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache, context::GpuProfilerLabel, kernels::DefaultKernelInvocable
};

pub struct BlitCopyOp;

struct BlitCopy<T: TensorElement> {
    dst: Tensor<T>,
    src: Tensor<T>,
    dst_offset: usize,
    src_offset: usize,
    size_bytes: usize,
    label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for BlitCopy<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_blit_encoder()?;
        let _scope = metallic_instrumentation::gpu_profiler::GpuProfiler::profile_blit(
            command_buffer.raw(),
            &encoder,
            self.label.op_name.clone(),
            self.label.backend.clone(),
        );
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &self.src.buf,
                self.src_offset,
                &self.dst.buf,
                self.dst_offset,
                self.size_bytes,
            );
        }
        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>) {
        // No compute args; this is a blit-only operation.
    }
}

impl DefaultKernelInvocable for BlitCopyOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, usize, Tensor<T>, usize, usize);

    fn function_id() -> Option<crate::kernels::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (dst, dst_offset, src, src_offset, size_bytes) = args;
        if size_bytes == 0 {
            return Err(MetalError::InvalidOperation("BlitCopyOp size_bytes == 0".into()));
        }
        ctx.prepare_tensors_for_active_cmd(&[&dst, &src])?;
        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("blit_copy"))
        } else {
            GpuProfilerLabel::fallback("blit_copy")
        };
        let op = BlitCopy {
            dst: dst.clone(),
            src: src.clone(),
            dst_offset,
            src_offset,
            size_bytes,
            label,
        };
        Ok((Box::new(op), dst))
    }
}
