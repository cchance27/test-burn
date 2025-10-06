use super::*;
use crate::context::GpuProfilerLabel;
use crate::{TensorElement, TensorInit, TensorStorage};
use metallic_instrumentation::GpuProfiler;

// User-facing struct for the broadcast element-wise add operation.
pub struct BroadcastElemwiseAddOp;

/// Broadcast add that writes the result back into the first operand. This avoids allocating a
/// fresh output tensor when the caller is willing to consume the input buffer.
pub struct BroadcastElemwiseAddInplaceOp;

// Internal struct that holds the operation data.
struct BroadcastElemwiseAdd<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    b_len: usize,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl KernelInvocable for BroadcastElemwiseAddOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseBroadcastAdd)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        let b_len = b.len();
        if b_len == 0 {
            return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
        }
        if b.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!("Broadcast b must be 1D, got {:?}", b.dims())));
        }

        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("broadcast_elemwise_add_op"));
        let op = BroadcastElemwiseAdd {
            a,
            b,
            out: out.clone(),
            b_len,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };
        Ok((Box::new(op), out))
    }
}

impl KernelInvocable for BroadcastElemwiseAddInplaceOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseBroadcastAdd)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        let b_len = b.len();
        if b_len == 0 {
            return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
        }
        if b.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!("Broadcast b must be 1D, got {:?}", b.dims())));
        }

        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

        let out = a.clone();

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("broadcast_elemwise_add_inplace_op"));

        let op = BroadcastElemwiseAddInplace {
            a,
            b,
            b_len,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for BroadcastElemwiseAdd<T> {
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
        set_bytes(&encoder, 4, &(self.b_len as u32));

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

struct BroadcastElemwiseAddInplace<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    b_len: usize,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for BroadcastElemwiseAddInplace<T> {
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
        set_buffer(&encoder, 2, &self.a.buf, self.a.offset);
        set_bytes(&encoder, 3, &total_elements);
        set_bytes(&encoder, 4, &(self.b_len as u32));

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
