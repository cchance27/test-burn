use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, context::GpuProfilerLabel, encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state}, operation::EncoderType, resource_cache::ResourceCache
};

pub struct SampleTopKMergeAndSampleOp;

struct SampleTopKMergeAndSample<U: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partials: Tensor<F32Element>,
    output_token: Tensor<U>,
    params: SampleParams,
    threads_per_tg: usize,
    profiler_label: GpuProfilerLabel,
}

impl<U: TensorElement> Operation for SampleTopKMergeAndSample<U> {
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
        set_buffer(&encoder, 0, &self.partials.buf, self.partials.offset);
        set_buffer(&encoder, 1, &self.output_token.buf, self.output_token.offset);
        set_bytes(&encoder, 2, &self.params);

        let tptg = self.threads_per_tg;
        let grid = MTLSize {
            width: tptg,
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

impl CustomKernelInvocable for SampleTopKMergeAndSampleOp {
    type Args<'a, T: TensorElement> = (Tensor<F32Element>, SampleParams);
    type OutputTensor<U: TensorElement> = U32;

    fn function_id() -> Option<KernelFunction> {
        KernelFunction::SampleTopKMergeAndSample.into()
    }

    fn new<'a, T: TensorElement, U: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<Self::OutputTensor<U>>), MetalError> {
        let (partials, params) = args;
        let pipeline = pipeline.ok_or(MetalError::InvalidOperation("Pipeline not provided".into()))?;

        let mut params = params;
        let (threads_per_tg, num_tgs) = calculate_threads_per_tg_and_num_threadgroups(&pipeline, params.vocab_size);
        params.num_threadgroups = num_tgs;
        
        let output_token = Tensor::zeros_of_type::<U32>(vec![1], ctx)?;
        let op = Box::new(SampleTopKMergeAndSample {
            pipeline,
            partials: partials.clone(),
            output_token: output_token.clone(),
            params,
            threads_per_tg,
            profiler_label: GpuProfilerLabel::new("sample_topk_merge_and_sample".into(), "Custom".into()),
        });

        Ok((op, output_token))
    }
}
