use super::*;

mod softmax_test;

/// Public, user-facing, zero-sized struct for the Softmax operation.
pub struct SoftmaxOp;

/// Internal struct that holds data for the Operation trait.
struct SoftmaxOperation {
    attn: Tensor,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    query_offset: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SoftmaxOp {
    type Args = (Tensor, u32, u32, u32, u32); // (attn, seq_q, seq_k, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedSoftmax)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (mut attn, seq_q, seq_k, causal, query_offset) = args;

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

        ctx.prepare_tensors_for_active_cmd(&mut [&mut attn]);

        let op = SoftmaxOperation {
            attn: attn.clone(),
            seq_q,
            seq_k,
            causal,
            query_offset,
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
        set_bytes(&encoder, 4, &self.query_offset);
        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
