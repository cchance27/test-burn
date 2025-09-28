use super::*;

pub struct RepeatKvHeadsOp;
pub struct RepeatKvHeadsStepOp;

struct RepeatKvHeads {
    input: Tensor,
    output: Tensor,
    group_size: u32,
    n_kv_heads: u32,
    n_heads: u32,
    seq: u32,
    head_dim: u32,
    total_elements: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for RepeatKvHeadsOp {
    type Args = (Tensor, u32, u32, u32, u32, u32, u32);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RepeatKvHeads)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (mut input, group_size, batch, n_kv_heads, n_heads, seq, head_dim) = args;

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

        let input_dims = input.dims();
        let batch_heads = (batch * n_kv_heads) as usize;
        if input_dims.len() != 3
            || input_dims[0] != seq as usize
            || input_dims[1] != batch_heads
            || input_dims[2] != head_dim as usize
        {
            return Err(MetalError::InvalidShape(format!(
                "Input dims {:?} must be [seq, batch*n_kv_heads, head_dim]",
                input_dims
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&mut [&mut input]);

        let output_dims = vec![(batch * n_heads) as usize, seq as usize, head_dim as usize];
        let output = Tensor::create_tensor_pooled(output_dims, ctx)?;
        let total_elements = output.len() as u32;

        let op = RepeatKvHeads {
            input,
            output: output.clone(),
            group_size,
            batch,
            n_kv_heads,
            n_heads,
            seq,
            head_dim,
            total_elements,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for RepeatKvHeads {
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
        set_bytes(&encoder, 8, &self.total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

struct RepeatKvHeadsStep {
    input: Tensor,
    output: Tensor,
    step: u32,
    group_size: u32,
    n_kv_heads: u32,
    n_heads: u32,
    total_batch_kv_heads: u32,
    total_batch_heads: u32,
    seq_len: u32,
    head_dim: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for RepeatKvHeadsStepOp {
    type Args = (Tensor, Tensor, u32, u32, u32, u32, u32, u32, u32, u32);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RepeatKvHeadsStep)
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (
            mut input,
            mut output,
            step,
            group_size,
            n_kv_heads,
            n_heads,
            total_batch_kv_heads,
            total_batch_heads,
            seq_len,
            head_dim,
        ) = args;

        if group_size == 0 {
            return Err(MetalError::InvalidShape(
                "group_size for repeat_kv_heads_step must be greater than zero".to_string(),
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
        if step >= seq_len {
            return Err(MetalError::InvalidShape(format!(
                "step {} must be less than seq_len {}",
                step, seq_len
            )));
        }

        let input_dims = input.dims();
        if input_dims.len() != 3
            || input_dims[0] != seq_len as usize
            || input_dims[1] != total_batch_kv_heads as usize
            || input_dims[2] != head_dim as usize
        {
            return Err(MetalError::InvalidShape(format!(
                "Input dims {:?} must be [seq_len, batch*n_kv_heads, head_dim]",
                input_dims
            )));
        }

        let output_dims = output.dims();
        if output_dims.len() != 3
            || output_dims[0] != total_batch_heads as usize
            || output_dims[1] != seq_len as usize
            || output_dims[2] != head_dim as usize
        {
            return Err(MetalError::InvalidShape(format!(
                "Output dims {:?} must be [batch*n_heads, seq_len, head_dim]",
                output_dims
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&mut [&mut input, &mut output]);

        let op = RepeatKvHeadsStep {
            input,
            output: output.clone(),
            step,
            group_size,
            n_kv_heads,
            n_heads,
            total_batch_kv_heads,
            total_batch_heads,
            seq_len,
            head_dim,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for RepeatKvHeadsStep {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = (self.total_batch_heads * self.head_dim) as u32;
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
        set_bytes(&encoder, 2, &self.step);
        set_bytes(&encoder, 3, &self.group_size);
        set_bytes(&encoder, 4, &self.n_kv_heads);
        set_bytes(&encoder, 5, &self.n_heads);
        set_bytes(&encoder, 6, &self.total_batch_kv_heads);
        set_bytes(&encoder, 7, &self.total_batch_heads);
        set_bytes(&encoder, 8, &self.seq_len);
        set_bytes(&encoder, 9, &self.head_dim);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

mod repeat_kv_heads_test;
