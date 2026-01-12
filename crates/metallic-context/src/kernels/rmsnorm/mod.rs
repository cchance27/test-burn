use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, operation::ComputeKernelEncoder};

pub struct RMSNormOp;

struct RMSNorm<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    gamma: Tensor<T>,
    feature_dim: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for RMSNormOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, u32); // (input, gamma, feature_dim)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RMSNorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, gamma, feature_dim) = args;

        // Validate dimensions
        if input.dims().last() != Some(&(feature_dim as usize)) {
            return Err(MetalError::InvalidShape(format!(
                "Input feature dimension {} does not match specified feature_dim {}",
                input.dims().last().unwrap_or(&0),
                feature_dim
            )));
        }
        if gamma.dims() != [feature_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Gamma shape {:?} does not match feature_dim {}",
                gamma.dims(),
                feature_dim
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&input, &gamma])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("rmsnorm_op"));

        let op = RMSNorm {
            input,
            output: output.clone(),
            gamma,
            feature_dim,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for RMSNorm<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_1d(self.input.len() as u32, 256);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        set_buffer(encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(encoder, 2, &self.gamma.buf, self.gamma.offset);
        set_bytes(encoder, 3, &self.feature_dim);
        set_bytes(encoder, 4, &(self.input.len() as u32));
    }
}

#[cfg(test)]
mod rmsnorm_test;
