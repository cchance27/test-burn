use super::*;

/// Public, user-facing, zero-sized struct for the SiLU operation.
pub struct SiluOp;

/// Internal struct that holds data for the Operation trait.
struct Silu {
    input: Tensor,
    output: Tensor,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SiluOp {
    type Args = Tensor;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Silu)
    }

    fn new(
        ctx: &mut Context,
        input: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let mut input = input;
        ctx.prepare_tensors_for_active_cmd(&mut [&mut input]);

        let output = Tensor::create_tensor_pooled(input.dims().to_vec(), ctx)?;

        let op = Silu {
            input,
            output: output.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for Silu {
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
        set_bytes(&encoder, 2, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod silu_test {
    use super::*;

    #[test]
    fn test_silu_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        let input_data = vec![1.0, -1.0, 0.0, 2.0];
        let input = Tensor::create_tensor_from_slice(&input_data, vec![4], &ctx)?;

        let result = ctx.call::<SiluOp>(input)?;

        // SiLU(x) = x * sigmoid(x)
        let expected: Vec<f32> = input_data.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect();
        let result_slice = result.as_slice();

        for (i, (result_val, expected_val)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert!(
                (result_val - expected_val).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                result_val,
                expected_val
            );
        }
        Ok(())
    }
}
