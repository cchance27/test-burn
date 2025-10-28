use metallic_instrumentation::{GpuProfiler, MetricEvent, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, Context, F32Element, MetalError, Operation, Tensor, TensorElement, context::GpuProfilerLabel, encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state}, operation::EncoderType, resource_cache::ResourceCache, tensor::dtypes::U32
};

pub struct SampleTopKPartialsOp;

pub struct SampleTopKPartials<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    input_logits: Tensor<T>,
    partials_values: Tensor<F32Element>,
    partials_indices: Tensor<U32>,
    params: SampleParams,
    threads_per_tg: usize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for SampleTopKPartials<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        command_buffer.prepare_encoder_for_operation(EncoderType::MetalCompute)?;
        let encoder = command_buffer.get_compute_encoder()?;
        let label = self.profiler_label.clone();
        let scope = GpuProfiler::profile_command_buffer(command_buffer.raw(), label.op_name.clone(), label.backend.clone());

        assert_ne!(self.params.num_threadgroups, 0, "num_threadgroups must be non-zero");
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input_logits.buf, self.input_logits.offset);
        set_buffer(&encoder, 1, &self.partials_values.buf, self.partials_values.offset);
        set_buffer(&encoder, 2, &self.partials_indices.buf, self.partials_indices.offset);
        set_bytes(&encoder, 3, &self.params);

        let tptg = self.threads_per_tg;
        let num_tgs = self.params.num_threadgroups as usize;

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
        let thread_groups = (
            u32::try_from(grid.width).unwrap_or(u32::MAX),
            u32::try_from(grid.height).unwrap_or(u32::MAX),
            u32::try_from(grid.depth).unwrap_or(u32::MAX),
        );
        record_metric_async!(MetricEvent::GpuKernelDispatched {
            kernel_name: "sample_topk_partials".to_string(),
            op_name: label.op_name.clone(),
            thread_groups,
        });

        dispatch_threads(&encoder, grid, tg);

        drop(scope);
        Ok(())
    }
}

impl CustomKernelInvocable for SampleTopKPartialsOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, f32, f32, u32, u32);
    type OutputTuple<T: TensorElement> = (F32Element, U32); // Two separate buffers

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SampleTopKPartials)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, <Self::OutputTuple<T> as MultiTensorOutput<T>>::Tensors), MetalError> {
        let (input_logits, vocab, k, top_p, temperature, seed, per_thread_m_clamp) = args;
        let pipeline = pipeline.ok_or(MetalError::InvalidOperation("Pipeline not provided".into()))?;

        let (threads_per_tg, num_tgs) = calculate_threads_per_tg_and_num_threadgroups(&pipeline, vocab);

        let override_m = std::env::var("METALLIC_SAMPLE_PER_THREAD_M")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v >= 1);
        let default_m = k.min(4).max(1);
        let per_thread_m = override_m.unwrap_or(default_m).min(per_thread_m_clamp).max(1);
        let params = SampleParams {
            vocab_size: vocab,
            k,
            top_p,
            temperature,
            seed,
            per_thread_m,
            num_threadgroups: num_tgs,
        };

        const TG_OUTPUT_K: u32 = 256;
        // Guarantee the partials buffers scale exactly with num_tgs so merge can infer it
        let total_elements = num_tgs * TG_OUTPUT_K;

        let partials_values = Tensor::zeros_of_type::<F32Element>(vec![total_elements as usize], ctx)?;

        let partials_indices = Tensor::zeros_of_type::<U32>(vec![total_elements as usize], ctx)?;

        let op = Box::new(SampleTopKPartials::<T> {
            pipeline,
            input_logits: input_logits.clone(),
            partials_values: partials_values.clone(),
            partials_indices: partials_indices.clone(),
            params,
            threads_per_tg,
            profiler_label: GpuProfilerLabel::new("sample_topk_partials".into(), "Custom".into()),
        });

        Ok((op, (partials_values, partials_indices)))
    }
}
