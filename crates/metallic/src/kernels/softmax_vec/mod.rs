use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{KernelFunction, KernelInvocable, ResourceCache, dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}
};

// Public, user-facing, zero-sized struct for the operation.
pub struct SoftmaxVecOp;

// Internal struct that holds data for the `Operation` trait.
struct SoftmaxVec<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
    rows_total: u32,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    query_offset: u32,
}

impl KernelInvocable for SoftmaxVecOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, u32, u32, u32, u32, u32); // (input, rows_total, seq_q, seq_k, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SoftmaxVec)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, rows_total, seq_q, seq_k, causal, query_offset) = args;

        // Validate dimensions
        if input.dims().len() < 2 {
            return Err(MetalError::InvalidShape(format!(
                "SoftmaxVec input must be at least 2D, got {:?}",
                input.dims()
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[input])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("softmax_vec_op"));

        let op = SoftmaxVec {
            input: input.clone(),
            output: output.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
            rows_total,
            seq_q,
            seq_k,
            causal,
            query_offset,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for SoftmaxVec<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);
        // Match legacy behavior: use native execution width (min 32) and dispatch rows on Y
        let native = self.pipeline.threadExecutionWidth();
        let width = if native < 32 { 32 } else { native };
        let threads_per_threadgroup = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: 1,
            height: self.rows_total as usize,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        // Kernel expects (seq_q, seq_k, causal, query_offset)
        set_bytes(&encoder, 2, &self.seq_q);
        set_bytes(&encoder, 3, &self.seq_k);
        set_bytes(&encoder, 4, &self.causal);
        set_bytes(&encoder, 5, &self.query_offset);

        dispatch_threadgroups(&encoder, threadgroups, threads_per_threadgroup);
        Ok(())
    }
}
