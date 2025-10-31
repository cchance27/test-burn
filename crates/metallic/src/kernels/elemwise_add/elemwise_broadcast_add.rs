use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, operation::{ComputeKernelEncoder}, context::GpuProfilerLabel};

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

impl DefaultKernelInvocable for BroadcastElemwiseAddOp {
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

impl DefaultKernelInvocable for BroadcastElemwiseAddInplaceOp {
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
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
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

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(groups, threads_per_tg);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(encoder, 3, &(self.a.len() as u32));
        set_bytes(encoder, 4, &(self.b_len as u32));
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
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
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

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(groups, threads_per_tg);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(encoder, 2, &self.a.buf, self.a.offset); // output is the same as input for inplace
        set_bytes(encoder, 3, &(self.a.len() as u32));
        set_bytes(encoder, 4, &(self.b_len as u32));
    }
}
