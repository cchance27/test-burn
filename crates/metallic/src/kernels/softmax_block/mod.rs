use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{KernelFunction, KernelInvocable, ResourceCache, dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}
};

// Public, user-facing, zero-sized struct for the operation.
pub struct SoftmaxBlockOp;

// Internal struct that holds data for the `Operation` trait.
struct SoftmaxBlock<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
    batch: u32,
    seq_q: u32,
    seq_k: u32,
    segment_size: u32,
    causal: u32,
    query_offset: u32,
}

impl KernelInvocable for SoftmaxBlockOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, u32, u32, u32, u32, u32, u32);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SoftmaxBlock)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, rows_total, seq_q, seq_k, _segment_size, causal, query_offset) = args;

        let batch = rows_total / seq_q;

        ctx.prepare_tensors_for_active_cmd(&[input])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("softmax_block_op"));

        // Calculate segment size (heuristic: aim for reasonable number of segments)
        let segment_size = 1024u32; // Fixed segment size for now

        let op = SoftmaxBlock {
            input: input.clone(),
            output: output.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
            batch,
            seq_q,
            seq_k,
            segment_size,
            causal,
            query_offset,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for SoftmaxBlock<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

        const THREADS_PER_TG: u32 = 256;
        const _SEGMENTS_PER_TG: u32 = 4; // TODO: not used?

        // Calculate number of segments needed
        let num_segments = self.seq_k.div_ceil(self.segment_size);
        let _rows_per_tg = 1; // TODO: One row per threadgroup for simplicity

        let threadgroups = MTLSize {
            width: (self.batch * self.seq_q) as usize,
            height: num_segments as usize,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: THREADS_PER_TG as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_bytes(&encoder, 2, &self.batch);
        set_bytes(&encoder, 3, &self.seq_q);
        set_bytes(&encoder, 4, &self.seq_k);
        set_bytes(&encoder, 5, &self.segment_size);
        set_bytes(&encoder, 6, &self.causal);
        set_bytes(&encoder, 7, &self.query_offset);

        dispatch_threadgroups(&encoder, threadgroups, threads_per_threadgroup);
        Ok(())
    }
}
