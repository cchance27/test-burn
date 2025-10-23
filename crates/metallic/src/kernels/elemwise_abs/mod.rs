use metallic_instrumentation::GpuProfiler;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel};

// Tests for Main Operation
#[cfg(test)]
mod elemwise_abs_test;

// User-facing struct for the standard element-wise absolute operation.
pub struct ElemwiseAbsOp;

// Internal struct that holds the operation data.
struct ElemwiseAbs<T: TensorElement> {
    a: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl KernelInvocable for ElemwiseAbsOp {
    type Args<'a, T: TensorElement> = (Tensor<T>,);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseAbs)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a,) = args;
        ctx.prepare_tensors_for_active_cmd(&[&a])?;
        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("elemwise_abs_op"));
        let op = ElemwiseAbs {
            a,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };
        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for ElemwiseAbs<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

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
        set_buffer(&encoder, 1, &self.out.buf, self.out.offset);
        set_bytes(&encoder, 2, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        drop(scope);
        Ok(())
    }
}
