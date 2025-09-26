use super::*;

// User-facing struct for the broadcast element-wise add operation.
pub struct BroadcastElemwiseAddOp;

// Internal struct that holds the operation data.
struct BroadcastElemwiseAdd {
    a: Tensor,
    b: Tensor,
    out: Tensor,
    b_len: usize,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for BroadcastElemwiseAddOp {
    type Args = (Tensor, Tensor);
    type Output = Tensor;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseBroadcastAdd)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        let (a, b) = args;
        let b_len = b.len();
        if b_len == 0 {
            return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
        }
        if b.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!("Broadcast b must be 1D, got {:?}", b.dims())));
        }

        let out = Tensor::create_tensor_pooled(a.dims().to_vec(), ctx)?;
        let op = BroadcastElemwiseAdd {
            a,
            b,
            out: out.clone(),
            b_len,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };
        Ok((Box::new(op), out))
    }
}

impl Operation for BroadcastElemwiseAdd {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.a.len() as u32;
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
        set_buffer(&encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(&encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(&encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(&encoder, 3, &total_elements);
        set_bytes(&encoder, 4, &(self.b_len as u32));

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
