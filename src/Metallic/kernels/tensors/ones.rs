use super::*;
use crate::metallic::encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct OnesOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct Ones {
    dims: Vec<usize>,
    out: Tensor,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// 3. Implement `KernelInvocable` for the public struct.
impl KernelInvocable for OnesOp {
    // Input arguments for the call.
    type Args = Vec<usize>;
    // The output type.
    type Output = Tensor;

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Ones)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new(
        ctx: &mut Context,
        dims: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        // Create the output tensor.
        let out = Tensor::create_tensor_pooled(dims.clone(), ctx)?;

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
impl Operation for Ones {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

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

        encoder.endEncoding();

        Ok(())
    }
}

#[cfg(test)]
mod ones_test {
    use crate::metallic::kernels::tensors::OnesOp;
    use crate::metallic::{Context, MetalError};

    #[test]
    fn test_ones() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        let result = ctx.call::<OnesOp>(vec![5])?;
        ctx.synchronize();

        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }
}
