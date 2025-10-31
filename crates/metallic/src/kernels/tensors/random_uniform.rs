use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{
    CommandBuffer, TensorElement, TensorInit, TensorStorage, operation::{ComputeKernelEncoder}, context::GpuProfilerLabel
};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct RandomUniformOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct RandomUniform<T: TensorElement> {
    dims: Vec<usize>,
    min_val: f32,
    max_val: f32,
    seed: u32,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

// 3. Implement `KernelInvocable` for the public struct.
impl DefaultKernelInvocable for RandomUniformOp {
    // Input arguments for the call.
    type Args<'a, T: TensorElement> = (Vec<usize>, f32, f32, Option<u32>);
    // The output type.

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RandomUniform)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (dims, min_val, max_val, seed_opt) = args;

        // Create the output tensor.
        let out = Tensor::new(dims.clone(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Generate or use provided seed
        let seed = seed_opt.unwrap_or_else(|| {
            let seed = ctx.rng_seed_counter as u32;
            ctx.rng_seed_counter += 1;
            seed
        });

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("random_uniform_op"));

        // Create the internal operation struct.
        let op = RandomUniform {
            dims,
            min_val,
            max_val,
            seed,
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
impl<T: TensorElement> Operation for RandomUniform<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        // Calculate total elements
        let total_elements: usize = self.dims.iter().product();

        // Create scale
        let _scale = self.max_val - self.min_val;

        // Dispatch threads - each thread handles 1 element
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: total_elements.div_ceil(threadgroup_size.width),
            height: 1,
            depth: 1,
        };

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(threadgroups, threadgroup_size);

        Ok(())
    }

    fn bind_to_encoder(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.out.buf, self.out.offset);
        set_bytes(encoder, 1, &self.seed);
        set_bytes(encoder, 2, &self.min_val);
        set_bytes(encoder, 3, &(self.max_val - self.min_val));
    }
}

#[cfg(test)]
mod random_uniform_test {
    use crate::{Context, F32Element, MetalError, kernels::tensors::RandomUniformOp};

    #[test]
    fn test_random_uniform() -> Result<(), MetalError> {
        let mut ctx = Context::<F32Element>::new()?;
        let result = ctx.call::<RandomUniformOp>((vec![5], 0.0, 1.0, Some(42)))?;

        let values = result.as_slice();
        assert_eq!(values.len(), 5);
        for &val in values {
            assert!((0.0..=1.0).contains(&val));
        }
        Ok(())
    }
}
