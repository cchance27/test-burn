use std::convert::TryFrom;

use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, operation::{ComputeKernelEncoder}};

/// Public, user-facing, zero-sized struct for the KV rearrange operation.
pub struct KvRearrangeOp;


#[cfg(test)]
mod tests;

/// Internal struct that holds data for the Operation trait.
struct KvRearrange<T: TensorElement> {
    input: Tensor<T>,  // [M, kv_dim]
    output: Tensor<T>, // [batch*n_heads, seq, head_dim]
    kv_dim: u32,
    row_stride: u32,
    kv_head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for KvRearrangeOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, u32, u32, u32, u32); // (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::KvRearrange)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq) = args;

        if input.dims().len() < 2 {
            return Err(MetalError::InvalidShape("KV rearrange expects at least 2D input".to_string()));
        }

        let row_stride_elems = input.strides.first().copied().unwrap_or_else(|| input.dims()[1]);
        if row_stride_elems == 0 {
            return Err(MetalError::InvalidShape(
                "Row stride for KV rearrange must be greater than zero".to_string(),
            ));
        }
        let row_stride =
            u32::try_from(row_stride_elems).map_err(|_| MetalError::InvalidShape("Row stride for KV rearrange exceeds u32".to_string()))?;

        // Calculate output dimensions: [batch*n_heads, seq, head_dim]
        // We need to infer batch from the input dimensions: M = batch*seq
        let input_m = input.dims()[0];
        let batch = input_m / seq as usize;
        let output_dims = vec![batch * n_heads as usize, seq as usize, head_dim as usize];

        ctx.prepare_tensors_for_active_cmd(&[&input])?;

        let output = Tensor::new(output_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("kv_rearrange_op"));

        let op = KvRearrange {
            input,
            output: output.clone(),
            kv_dim,
            row_stride,
            kv_head_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            seq,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for KvRearrange<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_1d(self.output.len() as u32, 256);
        
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(encoder, 1, &self.output.buf, self.output.offset);
        set_bytes(encoder, 2, &self.kv_dim);
        set_bytes(encoder, 3, &self.row_stride);
        set_bytes(encoder, 4, &self.kv_head_dim);
        set_bytes(encoder, 5, &self.n_heads);
        set_bytes(encoder, 6, &self.n_kv_heads);
        set_bytes(encoder, 7, &self.head_dim);
        set_bytes(encoder, 8, &self.seq);
        set_bytes(encoder, 9, &(self.output.len() as u32));
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn cpu_reference_rearrange(
    fused: &[f32],
    row_stride: usize,
    batch: usize,
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
    column_offset: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * n_heads * seq * head_dim];
    let group_size = n_heads / n_kv_heads;
    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / group_size;
            for s in 0..seq {
                let src_row = b * seq + s;
                for d in 0..head_dim {
                    let base_offset = kv_h * kv_head_dim + d;
                    let src_index = src_row * row_stride + column_offset + base_offset;
                    let dst_index = ((b * n_heads + h) * seq + s) * head_dim + d;
                    output[dst_index] = fused[src_index];
                }
            }
        }
    }
    output
}