use super::*;
use crate::metallic::{TensorElement, TensorInit, TensorStorage};

pub struct RepeatKvHeadsOp;

struct RepeatKvHeads<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    group_size: u32,
    batch: u32,
    n_kv_heads: u32,
    n_heads: u32,
    seq: u32,
    head_dim: u32,
    cache_stride: u32,
    total_elements: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for RepeatKvHeadsOp {
    #[allow(clippy::type_complexity)]
    type Args<'a, T: TensorElement> = (Tensor<T>, Option<Tensor<T>>, u32, u32, u32, u32, u32, u32, u32);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RepeatKvHeads)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, output_override, group_size, batch, n_kv_heads, n_heads, seq, head_dim, cache_stride) = args;

        if group_size == 0 {
            return Err(MetalError::InvalidShape(
                "group_size for repeat_kv_heads must be greater than zero".to_string(),
            ));
        }
        if n_heads == 0 || n_kv_heads == 0 {
            return Err(MetalError::InvalidShape(
                "Head counts for repeat_kv_heads must be greater than zero".to_string(),
            ));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(MetalError::InvalidShape(format!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                n_heads, n_kv_heads
            )));
        }
        if group_size != n_heads / n_kv_heads {
            return Err(MetalError::InvalidShape(format!(
                "group_size ({}) must equal n_heads / n_kv_heads ({})",
                group_size,
                n_heads / n_kv_heads
            )));
        }
        if seq == 0 {
            return Err(MetalError::InvalidShape(
                "Active sequence length for repeat_kv_heads must be greater than zero".to_string(),
            ));
        }

        let input_dims = input.dims();
        if input_dims.len() != 3 || input_dims[0] != (batch * n_kv_heads) as usize || input_dims[2] != head_dim as usize {
            return Err(MetalError::InvalidShape(format!(
                "Input dims {:?} must be [batch*n_kv_heads, seq, head_dim]",
                input_dims
            )));
        }
        if input_dims[1] != seq as usize {
            return Err(MetalError::InvalidShape(format!(
                "Input sequence (len={}) must match requested materialization seq ({})",
                input_dims[1], seq
            )));
        }

        let input_strides = input.strides.clone();
        if input_strides.len() < 2 {
            return Err(MetalError::InvalidShape(
                "Input tensor for repeat_kv_heads must expose at least two strides".to_string(),
            ));
        }
        let computed_stride = input_strides[0]
            .checked_div(head_dim as usize)
            .ok_or_else(|| MetalError::InvalidShape("Input tensor batch stride must be divisible by head_dim".to_string()))?;
        if cache_stride as usize != computed_stride {
            return Err(MetalError::InvalidShape(format!(
                "Provided cache stride ({}) does not match tensor batch stride ({}). Expected strides to agree for repeat_kv_heads",
                cache_stride, computed_stride
            )));
        }

        let output = if let Some(tensor) = output_override {
            tensor
        } else {
            let output_dims = vec![(batch * n_heads) as usize, seq as usize, head_dim as usize];
            Tensor::new(output_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?
        };

        let output_dims = output.dims().to_vec();
        if output_dims.len() != 3 || output_dims[0] != (batch * n_heads) as usize || output_dims[2] != head_dim as usize {
            return Err(MetalError::InvalidShape(format!(
                "Output dims {:?} must be [batch*n_heads, seq, head_dim]",
                output_dims
            )));
        }

        if output_dims[1] != seq as usize {
            return Err(MetalError::InvalidShape(format!(
                "Output sequence (len={}) must match requested materialization seq ({})",
                output_dims[1], seq
            )));
        }

        let output_strides = output.strides.clone();
        if output_strides.len() < 2 {
            return Err(MetalError::InvalidShape(
                "Output tensor for repeat_kv_heads must expose at least two strides".to_string(),
            ));
        }

        if output_strides[2] != 1 {
            return Err(MetalError::InvalidShape(
                "Output tensor for repeat_kv_heads must have contiguous head_dim stride".to_string(),
            ));
        }

        ctx.prepare_tensors_for_active_cmd(&[&input, &output])?;

        let total_elements_u64 = (batch as u64)
            .checked_mul(n_heads as u64)
            .and_then(|v| v.checked_mul(seq as u64))
            .and_then(|v| v.checked_mul(head_dim as u64))
            .ok_or_else(|| MetalError::InvalidShape("repeat_kv_heads element count overflowed u32".to_string()))?;

        if total_elements_u64 == 0 {
            return Err(MetalError::InvalidShape(
                "repeat_kv_heads requires a non-zero element count".to_string(),
            ));
        }

        let total_elements = total_elements_u64 as u32;

        let op = RepeatKvHeads {
            input,
            output: output.clone(),
            group_size,
            batch,
            n_kv_heads,
            n_heads,
            seq,
            head_dim,
            cache_stride: computed_stride as u32,
            total_elements,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for RepeatKvHeads<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: self.total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_bytes(&encoder, 2, &self.group_size);
        set_bytes(&encoder, 3, &self.batch);
        set_bytes(&encoder, 4, &self.n_kv_heads);
        set_bytes(&encoder, 5, &self.n_heads);
        set_bytes(&encoder, 6, &self.seq);
        set_bytes(&encoder, 7, &self.head_dim);
        set_bytes(&encoder, 8, &self.cache_stride);
        set_bytes(&encoder, 9, &self.total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod repeat_kv_heads_test;
