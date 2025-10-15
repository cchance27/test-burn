use super::*;
use crate::encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state};
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct OnesOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct Ones<T: TensorElement> {
    dims: Vec<usize>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// 3. Implement `KernelInvocable` for the public struct.
impl KernelInvocable for OnesOp {
    // Input arguments for the call.
    type Args<'a, T: TensorElement> = Vec<usize>;
    // The output type.

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Ones)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        dims: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Create the output tensor.
        let out = Tensor::new(dims.clone(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Create the internal operation struct.
        let op = Ones {
            dims,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// 4. Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl<T: TensorElement> Operation for Ones<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        // Calculate total elements
        let total_elements: usize = self.dims.iter().product();

        // Create and set the constant buffer for total_elements
        let total_elements_u32 = total_elements as u32;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.out.buf, self.out.offset);
        set_bytes(&encoder, 1, &total_elements_u32);

        // Dispatch threads - each thread handles 4 elements
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: total_elements.div_ceil(4).div_ceil(threadgroup_size.width),
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(&encoder, threadgroups, threadgroup_size);

        Ok(())
    }
}

#[cfg(test)]
mod ones_test {
    use crate::kernels::tensors::OnesOp;
    use crate::{Context, F32Element, MetalError};

    #[test]
    fn test_ones() -> Result<(), MetalError> {
        let mut ctx: Context<F32Element> = Context::<F32Element>::new()?;
        let result = ctx.call::<OnesOp>(vec![5])?;

        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }
}
