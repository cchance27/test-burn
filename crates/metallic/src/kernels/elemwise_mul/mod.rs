use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage};

pub struct ElemwiseMulOp;

#[cfg(test)]
mod elemwise_mul_test;

struct ElemwiseMul<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl DefaultKernelInvocable for ElemwiseMulOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseMul)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        if a.dims() != b.dims() {
            return Err(MetalError::InvalidShape(format!(
                "ElemwiseMul: input shapes must match, got a={:?}, b={:?}",
                a.dims(),
                b.dims(),
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let op = ElemwiseMul {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for ElemwiseMul<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

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
        Ok(())
    }
}
