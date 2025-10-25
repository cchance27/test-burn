use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    context::GpuProfilerLabel, encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state}, operation::EncoderType, resource_cache::ResourceCache, CommandBuffer, Context, F32Element, MetalError, Operation, Tensor, TensorElement 
};

pub struct SampleTopKPartialsOp;

pub struct SampleTopKPartials<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    input_logits: Tensor<T>,
    partials: Tensor<F32Element>, // Does this need to be F32?
    params: SampleParams,
    threads_per_tg: usize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for SampleTopKPartials<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        command_buffer.prepare_encoder_for_operation(EncoderType::MetalCompute)?;
        let encoder = command_buffer.get_compute_encoder()?;
        let scope = GpuProfiler::profile_command_buffer(
            command_buffer.raw(),
            self.profiler_label.op_name.clone(),
            self.profiler_label.backend.clone(),
        );

        assert_ne!(self.params.num_threadgroups, 0, "num_threadgroups must be non-zero, set via new()");
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input_logits.buf, self.input_logits.offset);
        set_buffer(&encoder, 1, &self.partials.buf, self.partials.offset);
        set_bytes(&encoder, 2, &self.params);

        let vocab = self.params.vocab_size as usize;
        let tptg = self.threads_per_tg;
        let num_tgs = vocab.div_ceil(tptg);

        let grid = MTLSize {
            width: (num_tgs * tptg),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tptg,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid, tg);

        drop(scope);
        Ok(())
    }
}

impl CustomKernelInvocable for SampleTopKPartialsOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, f32, f32, u32, u32);
    type OutputTensor<U: TensorElement> = F32Element;

    fn function_id() -> Option<KernelFunction> {
        // Use the input tensor dtype (T) to determine which kernel function to use
        // Since this kernel processes the input logits, it needs to match the input dtype
        Some(KernelFunction::SampleTopKPartials)
    }

    fn new<'a, T: TensorElement, U: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<Self::OutputTensor<U>>), MetalError> {
        let (input_logits, vocab, k, top_p, temperature, seed, per_thread_m_clamp) = args;
        let pipeline = pipeline.ok_or(MetalError::InvalidOperation("Pipeline not provided".into()))?;

        let (threads_per_tg, num_tgs) = calculate_threads_per_tg_and_num_threadgroups(&pipeline, vocab);

        // Choose per-thread M based on requested top-k, capped to a small constant for shared memory/local array bounds.
        let per_thread_m = k.clamp(1, per_thread_m_clamp); // trim partial device writes: per-thread M = min(k,16) // we had this set to 16 previously but it was slower
        let params = SampleParams {
            vocab_size: vocab,
            k,
            top_p,
            temperature,
            seed,
            per_thread_m,
            num_threadgroups: num_tgs,
        };

        let total_pairs = num_tgs * threads_per_tg as u32 * per_thread_m; // per_thread_m elements per thread
        let partials = Tensor::zeros_of_type::<F32Element>(vec![total_pairs as usize * 2], ctx)?; // 2 components per LogitPair (float, uint) // Does this need to be F32?

        let op = Box::new(SampleTopKPartials::<T> {
            pipeline,
            input_logits: input_logits.clone(),
            partials: partials.clone(),
            params,
            threads_per_tg,
            profiler_label: GpuProfilerLabel::new("sample_topk_partials".into(), "Custom".into()),
        });

        Ok((op, partials))
    }
}
