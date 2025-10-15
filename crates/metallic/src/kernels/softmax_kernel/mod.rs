use super::*;
use crate::context::GpuProfilerLabel;
use crate::resource_cache::ResourceCache;

use crate::TensorElement;
use metallic_instrumentation::GpuProfiler;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

/// Public, user-facing, zero-sized struct for the legacy Softmax operation.
pub struct SoftmaxKernelOp;

/// Internal struct that holds data for the Operation trait.
struct SoftmaxKernelOperation<T: TensorElement> {
    attn: Tensor<T>,
    rows_total: u32,
    seq_q: u32,
    seq_k: u32,
    causal: u32,
    query_offset: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl KernelInvocable for SoftmaxKernelOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, u32, u32, u32, u32, u32); // (attn, rows_total, seq_q, seq_k, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedSoftmax)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (attn, rows_total, seq_q, seq_k, causal, query_offset) = args;

        // Validate dimensions
        if attn.dims().len() < 2 {
            return Err(MetalError::InvalidShape(format!(
                "Softmax input must be at least 2D, got {:?}",
                attn.dims()
            )));
        }

        let view = attn.as_mps_matrix_batch_view()?;
        if view.rows != seq_q as usize || view.columns != seq_k as usize {
            return Err(MetalError::InvalidShape(format!(
                "Attention matrix dimensions {:?} don't match seq_q={} x seq_k={}",
                attn.dims(),
                seq_q,
                seq_k
            )));
        }

        if rows_total as usize != view.batch * view.rows {
            return Err(MetalError::InvalidShape(format!(
                "Softmax rows_total {} does not match view layout {:?}",
                rows_total,
                attn.dims()
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[attn])?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("softmax_kernel_op"));

        let op = SoftmaxKernelOperation {
            attn: attn.clone(),
            rows_total,
            seq_q,
            seq_k,
            causal,
            query_offset,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), attn.clone())) // Return a shallow clone since operation is in-place
    }
}

impl<T: TensorElement> Operation for SoftmaxKernelOperation<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer, &encoder, label.op_name, label.backend);

        // Ensure at least 32 threads per threadgroup to satisfy kernel's reduction assumptions
        let native = self.pipeline.threadExecutionWidth();
        let width = if native < 32 { 32 } else { native };
        let threads_per_tg = MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: 1,
            height: self.rows_total as usize,
            depth: 1,
        };
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.attn.buf, self.attn.offset);
        set_bytes(&encoder, 1, &self.seq_q);
        set_bytes(&encoder, 2, &self.seq_k);
        set_bytes(&encoder, 3, &self.causal);
        set_bytes(&encoder, 4, &self.query_offset);
        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
