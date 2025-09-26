use super::*;

pub struct ElemwiseSubOp;

mod elemwise_sub_test;

struct ElemwiseSub {
    a: Tensor,
    b: Tensor,
    out: Tensor,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for ElemwiseSubOp {
    type Args = (Tensor, Tensor);
    type Output = Tensor;

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseSub)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        let (a, b) = args;
        if a.dims() != b.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseSub: input shapes must match, got a={:?}, b={:?}",
                a.dims(),
                b.dims(),
            )));
        }

        let out = Tensor::create_tensor_pooled(a.dims().to_vec(), ctx)?;

        let op = ElemwiseSub {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), out))
    }
}

impl Operation for ElemwiseSub {
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

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
