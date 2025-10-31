use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, caching::ResourceCache, context::GpuProfilerLabel, encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}
};

/// Public, user-facing, zero-sized struct for the RoPE operation.
pub struct RoPEOp;

/// Internal struct that holds data for the `Operation` trait.
struct RoPE<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    cos: Tensor<T>,
    sin: Tensor<T>,
    dim: u32,
    seq_len: u32,
    position_offset: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for RoPEOp {
    /// Input arguments for the call: (input, cos, sin, dim, seq_len, position_offset)
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, Tensor<T>, u32, u32, u32);

    /// Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Rope)
    }

    /// This `new` method is called by `ctx.call()`.
    /// It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, cos, sin, dim, seq_len, position_offset) = args;

        // Basic validation
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(MetalError::InvalidShape(format!(
                "dim must be positive and even for RoPE, got {}",
                dim
            )));
        }

        if cos.dims().len() != 2 || cos.dims()[1] != (dim as usize) / 2 {
            return Err(MetalError::InvalidShape(format!(
                "cos shape {:?} must be [N, dim/2] with N >= position_offset + seq_len (offset={}, seq_len={})",
                cos.dims(),
                position_offset,
                seq_len
            )));
        }
        if sin.dims().len() != 2 || sin.dims()[1] != (dim as usize) / 2 {
            return Err(MetalError::InvalidShape(format!(
                "sin shape {:?} must be [N, dim/2] with N >= position_offset + seq_len (offset={}, seq_len={})",
                sin.dims(),
                position_offset,
                seq_len
            )));
        }

        let cos_rows = cos.dims()[0];
        let sin_rows = sin.dims()[0];
        let required_rows = position_offset as usize + seq_len as usize;
        if cos_rows < required_rows || sin_rows < required_rows {
            return Err(MetalError::InvalidShape(format!(
                "RoPE caches require at least {} rows, got cos={} sin={}",
                required_rows, cos_rows, sin_rows
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&input, &cos, &sin])?;

        // Create the output tensor with same shape as input
        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("rope_op"));

        // Create the internal operation struct.
        let op = RoPE {
            input,
            output: output.clone(),
            cos,
            sin,
            dim,
            seq_len,
            position_offset,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for RoPE<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

        let total_elements = self.input.len() as u32;
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
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.cos.buf, self.cos.offset);
        set_buffer(&encoder, 3, &self.sin.buf, self.sin.offset);
        set_bytes(&encoder, 4, &self.dim);
        set_bytes(&encoder, 5, &self.seq_len);
        set_bytes(&encoder, 6, &self.position_offset);
        set_bytes(&encoder, 7, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        Ok(())
    }
}

#[cfg(test)]
mod rope_test;
