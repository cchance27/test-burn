use super::*;

/// Public, user-facing, zero-sized struct for the Softmax operation.
pub struct SoftmaxOp;

/// Internal struct that holds data for the Operation trait.
struct SoftmaxOperation {
    attn: Tensor,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SoftmaxOp {
    type Args = (Tensor, u32, u32, u32); // (attn, seq_q, seq_k, causal)
    type Output = Tensor; // Returns the same tensor that was modified in-place

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedSoftmax)
    }

    fn new(
        _ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        let (attn, seq_q, seq_k, causal) = args;

        // Validate dimensions
        if attn.dims().len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Softmax input must be 2D [seq_q, seq_k], got {:?}",
                attn.dims()
            )));
        }

        let expected_dims = [seq_q as usize, seq_k as usize];
        if attn.dims() != expected_dims {
            return Err(MetalError::InvalidShape(format!(
                "Attention matrix dimensions {:?} don't match seq_q={} x seq_k={}",
                attn.dims(),
                seq_q,
                seq_k
            )));
        }

        let op = SoftmaxOperation {
            attn: attn.clone(),
            seq_q,
            seq_k,
            causal,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), attn)) // Return the same tensor since operation is in-place
    }
}

impl Operation for SoftmaxOperation {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        // Ensure at least 32 threads per threadgroup to satisfy kernel's reduction assumptions
        let native = self.pipeline.threadExecutionWidth();
        let width = if native < 32 { 32 } else { native };
        let threads_per_tg = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: 1,
            height: self.seq_q as usize,
            depth: 1,
        };
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.attn.buf, self.attn.offset);
        set_bytes(&encoder, 1, &self.seq_q);
        set_bytes(&encoder, 2, &self.seq_k);
        set_bytes(&encoder, 3, &self.causal);
        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod softmax_test {
    use super::*;

    #[test]
    fn test_softmax_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        // Create a simple test tensor [2, 3] with values that will produce recognizable softmax results
        let input_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // Two rows to softmax independently
        let attn = Tensor::create_tensor_from_slice(&input_data, vec![2, 3], &ctx)?;

        // Apply softmax with no causal masking (causal=0)
        let result = ctx.call::<SoftmaxOp>((attn, 2, 3, 0))?;
        ctx.synchronize();

        // Check that each row sums to approximately 1 (property of softmax)
        let result_slice = result.as_slice();
        let row1_sum: f32 = result_slice[0..3].iter().sum();
        let row2_sum: f32 = result_slice[3..6].iter().sum();

        assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 sum should be 1.0, got {}", row1_sum);
        assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 sum should be 1.0, got {}", row2_sum);
        Ok(())
    }
}
