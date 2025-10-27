use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, Context, F32Element, MetalError, Operation, Tensor, TensorElement, context::GpuProfilerLabel, encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state}, operation::EncoderType, resource_cache::ResourceCache, tensor::dtypes::U32
};

pub struct SampleTopKMergeAndSampleOp;

struct SampleTopKMergeAndSample {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partials_values: Tensor<F32Element>,
    partials_indices: Tensor<U32>,
    output_token: Tensor<U32>,
    params: SampleParams,
    threads_per_tg: usize,
    profiler_label: GpuProfilerLabel,
}

impl Operation for SampleTopKMergeAndSample {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        command_buffer.prepare_encoder_for_operation(EncoderType::MetalCompute)?;
        let encoder = command_buffer.get_compute_encoder()?;
        let scope = GpuProfiler::profile_command_buffer(
            command_buffer.raw(),
            self.profiler_label.op_name.clone(),
            self.profiler_label.backend.clone(),
        );

        assert_ne!(self.params.num_threadgroups, 0, "num_threadgroups must be non-zero");
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.partials_values.buf, self.partials_values.offset);
        set_buffer(&encoder, 1, &self.partials_indices.buf, self.partials_indices.offset);
        set_buffer(&encoder, 2, &self.output_token.buf, self.output_token.offset);
        set_bytes(&encoder, 3, &self.params);

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
    type Args<'a, T: TensorElement> = (Tensor<F32Element>, Tensor<U32>, SampleParams);
    type OutputTuple<T: TensorElement> = (U32,);

    fn function_id() -> Option<KernelFunction> {
        KernelFunction::SampleTopKMergeAndSample.into()
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, <Self::OutputTuple<T> as MultiTensorOutput<T>>::Tensors), MetalError> {
        let (partials_values, partials_indices, params) = args;
        let pipeline = pipeline.ok_or(MetalError::InvalidOperation("Pipeline not provided".into()))?;

        let mut params = params;
        let (threads_per_tg, _num_tgs_calc) = calculate_threads_per_tg_and_num_threadgroups(&pipeline, params.vocab_size);
        // Derive the number of threadgroups from the partials buffer size to ensure consistency
        const TG_OUTPUT_K: u32 = 256;
        let derived_num_tgs = (partials_values.len() as u32)
            .checked_div(TG_OUTPUT_K)
            .ok_or(MetalError::InvalidOperation("Partials buffer too small".into()))?;
        params.num_threadgroups = derived_num_tgs.max(1);

        let output_token = Tensor::zeros_of_type::<U32>(vec![1], ctx)?;
        let op = Box::new(SampleTopKMergeAndSample {
            pipeline,
            partials_values: partials_values.clone(),
            partials_indices: partials_indices.clone(),
            output_token: output_token.clone(),
            params,
            threads_per_tg,
            profiler_label: GpuProfilerLabel::new("sample_topk_merge_and_sample".into(), "Custom".into()),
        });

        Ok((op, (output_token,)))
    }
}
