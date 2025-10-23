use super::*;
use crate::{
    CommandBuffer, TensorElement, TensorInit, TensorStorage, encoder::{dispatch_threadgroups, set_buffer, set_compute_pipeline_state}
};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct ArangeOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct Arange<T: TensorElement> {
    length: usize,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// 3. Implement `KernelInvocable` for the public struct.
impl KernelInvocable for ArangeOp {
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
        _cache: std::option::Option<&mut crate::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Create the output tensor.
        let out = Tensor::new(vec![length], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Create the internal operation struct.
        let op = Arange {
            length,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// 4. Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl<T: TensorElement> Operation for Arange<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.out.buf, self.out.offset);

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: self.length.div_ceil(threadgroup_size.width),
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(&encoder, threadgroups, threadgroup_size);

        Ok(())
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
