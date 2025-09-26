use super::*;

pub struct RMSNormOp;

struct RMSNorm {
    input: Tensor,
    output: Tensor,
    gamma: Tensor,
    feature_dim: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for RMSNormOp {
    type Args = (Tensor, Tensor, u32); // (input, gamma, feature_dim)
    type Output = Tensor;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RMSNorm)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
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

        let output = Tensor::create_tensor_pooled(input.dims().to_vec(), ctx)?;

        let op = RMSNorm {
            input,
            output: output.clone(),
            gamma,
            feature_dim,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for RMSNorm {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.input.len() as u32;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.gamma.buf, self.gamma.offset);
        set_bytes(&encoder, 3, &self.feature_dim);
        set_bytes(&encoder, 4, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

mod rmsnorm_test;
