use std::time::Instant;

use metallic_instrumentation::{GpuProfiler, MetricEvent, record_metric_async};
use objc2_metal::MTLResource;

use super::*;
use crate::{
    CommandBuffer, TensorElement, context::{GpuProfilerLabel, RepeatKvWorkspaceKind}
};

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
    profiler_label: GpuProfilerLabel,
}

impl KernelInvocable for RepeatKvHeadsOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, u32, u32, u32, u32, u32, u32, RepeatKvWorkspaceKind, bool);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::RepeatKvHeads)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, group_size, batch, n_kv_heads, n_heads, seq, head_dim, cache_stride, layer_idx, workspace_kind, prefer_shared) = args;

        // Graph backend path: build and encode MPSGraph repeat to avoid compute kernel
        let selection = ctx.backend_registry().select_sdpa(super::KernelBackendKind::Legacy);
        if selection.backend == super::KernelBackendKind::Graph {
            let repeated_heads = (batch as usize) * (n_heads as usize);
            let output = ctx.acquire_repeat_kv_workspace(
                layer_idx as usize,
                workspace_kind,
                repeated_heads,
                seq as usize,
                cache_stride as usize,
                head_dim as usize,
                true,
            )?;
            ctx.prepare_tensors_for_active_cmd(&[&input, &output])?;
            let (op, result) = super::repeat_kv_heads_graph::RepeatKvHeadsGraphOp::new(
                ctx,
                (
                    input.clone(),
                    output.clone(),
                    group_size,
                    batch,
                    n_kv_heads,
                    n_heads,
                    seq,
                    head_dim,
                    cache_stride,
                ),
                None,
                cache,
            )?;
            return Ok((op, result));
        }

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
        if seq > cache_stride {
            return Err(MetalError::InvalidShape(format!(
                "Active sequence length ({}) exceeds cache stride ({})",
                seq, cache_stride
            )));
        }

        let batch_usize =
            usize::try_from(batch).map_err(|_| MetalError::InvalidShape("repeat_kv_heads batch exceeds platform usize".to_string()))?;
        let n_kv_heads_usize = usize::try_from(n_kv_heads)
            .map_err(|_| MetalError::InvalidShape("repeat_kv_heads n_kv_heads exceeds platform usize".to_string()))?;
        let n_heads_usize =
            usize::try_from(n_heads).map_err(|_| MetalError::InvalidShape("repeat_kv_heads n_heads exceeds platform usize".to_string()))?;
        let seq_usize = usize::try_from(seq)
            .map_err(|_| MetalError::InvalidShape("repeat_kv_heads sequence length exceeds platform usize".to_string()))?;
        let head_dim_usize = usize::try_from(head_dim)
            .map_err(|_| MetalError::InvalidShape("repeat_kv_heads head_dim exceeds platform usize".to_string()))?;
        let cache_capacity = usize::try_from(cache_stride)
            .map_err(|_| MetalError::InvalidShape("repeat_kv_heads cache stride exceeds platform usize".to_string()))?;
        let layer_idx_usize = usize::try_from(layer_idx)
            .map_err(|_| MetalError::InvalidShape("repeat_kv_heads layer index exceeds platform usize".to_string()))?;

        let input_dims = input.dims();
        if input_dims.len() != 3 || input_dims[0] != batch_usize * n_kv_heads_usize || input_dims[2] != head_dim_usize {
            return Err(MetalError::InvalidShape(format!(
                "Input dims {:?} must be [batch*n_kv_heads, seq, head_dim]",
                input_dims
            )));
        }
        if input_dims[1] != seq_usize {
            return Err(MetalError::InvalidShape(format!(
                "Input sequence ({}) must match active sequence ({})",
                input_dims[1], seq
            )));
        }

        let input_strides = input.strides.clone();
        if input_strides.len() < 2 {
            return Err(MetalError::InvalidShape(
                "Input tensor for repeat_kv_heads must expose at least two strides".to_string(),
            ));
        }
        let expected_stride = cache_capacity * head_dim_usize;
        if input_strides[0] != expected_stride {
            return Err(MetalError::InvalidShape(format!(
                "Input batch stride ({}) does not match cache stride ({})",
                input_strides[0], expected_stride
            )));
        }

        let prep_input_start = Instant::now();
        ctx.prepare_tensors_for_active_cmd(&[&input])?;
        let prep_input_elapsed = prep_input_start.elapsed();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "tensor_prepare".to_string(),
            internal_kernel_name: format!(
                "repeat_kv/input/{:?}/storage={:?}/bytes={}",
                workspace_kind,
                input.buf.storageMode(),
                input.size_bytes()
            ),
            duration_us: (prep_input_elapsed.as_micros().max(1)) as u64,
        });

        let repeated_heads = batch_usize * n_heads_usize;
        let output = ctx.acquire_repeat_kv_workspace(
            layer_idx_usize,
            workspace_kind,
            repeated_heads,
            seq_usize,
            cache_capacity,
            head_dim_usize,
            prefer_shared,
        )?;
        let total_elements = output.len() as u32;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("repeat_kv_heads_op"));

        let op = RepeatKvHeads {
            input,
            output: output.clone(),
            group_size,
            batch,
            n_kv_heads,
            n_heads,
            seq,
            head_dim,
            cache_stride,
            total_elements,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for RepeatKvHeads<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

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
        Ok(())
    }
}

#[cfg(test)]
mod repeat_kv_heads_test;
