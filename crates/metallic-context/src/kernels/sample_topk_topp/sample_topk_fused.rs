use metallic_instrumentation::{MetricEvent, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLResourceOptions, MTLSize};

use super::*;
use crate::{
    CommandBuffer, Context, MetalError, Tensor, TensorElement, caching::ResourceCache, context::GpuProfilerLabel, operation::{ComputeKernelEncoder, EncoderType}, tensor::dtypes::U32
};

pub struct SampleTopKFusedOp;

pub struct SampleTopKFused<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    input_logits: Tensor<T>,
    output_token: Tensor<U32>,
    params: SampleParams,
    threads_per_tg: usize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for SampleTopKFused<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        command_buffer.prepare_encoder_for_operation(EncoderType::MetalCompute)?;

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
        let thread_groups = (
            u32::try_from(grid.width).unwrap_or(u32::MAX),
            u32::try_from(grid.height).unwrap_or(u32::MAX),
            u32::try_from(grid.depth).unwrap_or(u32::MAX),
        );
        record_metric_async!(MetricEvent::GpuKernelDispatched {
            kernel_name: "sample_topk_fused".to_string(),
            op_name: self.profiler_label.op_name.clone(),
            thread_groups,
        });

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_threads(grid, tg);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        set_buffer(encoder, 0, &self.input_logits.buf, self.input_logits.offset);
        set_buffer(encoder, 1, &self.output_token.buf, self.output_token.offset);
        set_bytes(encoder, 2, &self.params);
    }
}

fn select_threads_per_tg(pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> usize {
    const PREFERRED_THREADGROUPS: &[usize] = &[1024, 896, 768, 640, 512, 384, 352, 320, 256, 192, 128];

    let tew = pipeline.threadExecutionWidth().max(1);
    let max_total = pipeline.maxTotalThreadsPerThreadgroup();
    let max_tptg: usize = max_total.max(tew);
    let override_tptg = std::env::var("METALLIC_SAMPLE_TPTG")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0 && v <= max_tptg)
        .filter(|&v| v % tew == 0);

    if let Some(v) = override_tptg {
        return v;
    }

    for &candidate in PREFERRED_THREADGROUPS {
        if candidate <= max_tptg && candidate % tew == 0 {
            return candidate;
        }
    }

    let fallback = (max_total / tew).max(1) * tew;
    fallback.min(max_tptg)
}

fn select_default_per_thread_m(threads_per_tg: usize, k: u32, clamp: u32) -> u32 {
    let recommended = if threads_per_tg >= 1024 {
        6
    } else if threads_per_tg >= 896 {
        4
    } else if threads_per_tg >= 320 {
        6
    } else if threads_per_tg >= 256 {
        5
    } else if threads_per_tg >= 192 {
        4
    } else {
        3
    };

    recommended.min(k).min(clamp).max(1)
}

impl CustomKernelInvocable for SampleTopKFusedOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, f32, f32, u32, u32);
    type OutputTuple<T: TensorElement> = (U32,);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SampleTopKFused)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, <Self::OutputTuple<T> as MultiTensorOutput<T>>::Tensors), MetalError> {
        let (input_logits, vocab, k, top_p, temperature, seed, per_thread_m_clamp) = args;
        let pipeline = pipeline.ok_or(MetalError::InvalidOperation("Pipeline not provided".into()))?;

        let threads_per_tg = select_threads_per_tg(&pipeline);

        let override_m = std::env::var("METALLIC_SAMPLE_PER_THREAD_M")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v >= 1);
        let default_m = select_default_per_thread_m(threads_per_tg, k, per_thread_m_clamp);
        let per_thread_m = override_m.unwrap_or(default_m).min(per_thread_m_clamp).max(1);

        let params = SampleParams {
            vocab_size: vocab,
            k,
            top_p,
            temperature,
            seed,
            per_thread_m,
            num_threadgroups: 1,
        };

        let buffer_len = std::mem::size_of::<u32>();
        let buffer = ctx
            .device
            .newBufferWithLength_options(buffer_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(buffer_len))?;
        let mut output_token = Tensor::<U32>::from_existing_buffer(buffer, vec![1], U32::DTYPE, &ctx.device, &ctx.command_queue, 0, true)?;
        output_token.as_mut_slice()[0] = 0;

        let op = Box::new(SampleTopKFused::<T> {
            pipeline,
            input_logits: input_logits.clone(),
            output_token: output_token.clone(),
            params,
            threads_per_tg,
            profiler_label: GpuProfilerLabel::new("sample_topk_fused".into(), "Custom".into()),
        });

        Ok((op, (output_token,)))
    }
}
