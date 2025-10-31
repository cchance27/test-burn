use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{
    CommandBuffer, TensorElement, TensorInit, TensorStorage, operation::{ComputeKernelEncoder}, context::GpuProfilerLabel
};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct ArangeOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct Arange<T: TensorElement> {
    length: usize,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

// 3. Implement `KernelInvocable` for the public struct.
impl DefaultKernelInvocable for ArangeOp {
    // Input arguments for the call.
    type Args<'a, T: TensorElement> = usize;
    // The output type.

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Arange)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        length: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Create the output tensor.
        let out = Tensor::new(vec![length], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("arange_op"));

        // Create the internal operation struct.
        let op = Arange {
            length,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// 4. Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl<T: TensorElement> Operation for Arange<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_1d(self.length as u32, 256);
        
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::set_buffer;
        
        set_buffer(encoder, 0, &self.out.buf, self.out.offset);
    }
}

#[cfg(test)]
mod arange_test {
    use crate::{Context, F32Element, MetalError, kernels::tensors::ArangeOp};

    #[test]
    fn test_arange() -> Result<(), MetalError> {
        let mut ctx = Context::<F32Element>::new()?;
        let result = ctx.call::<ArangeOp>(5)?;

        assert_eq!(result.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
}
