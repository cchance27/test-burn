use objc2_metal::MTLComputeCommandEncoder;

use super::*;
use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache, context::GpuProfilerLabel, operation::ComputeKernelEncoder, tensor::{TensorInit, TensorStorage, dtypes::U32}
};

/// GPU embedding lookup: gather rows for given token ids into an output [batch, seq, d_model].
pub struct EmbeddingLookupOp;

struct EmbeddingLookup<T: TensorElement> {
    table: Tensor<T>,
    indices: Tensor<U32>,
    out: Tensor<T>,
    d_model: u32,
    total: u32,
    vocab: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for EmbeddingLookupOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<U32>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::EmbeddingLookup)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (table, indices) = args;

        // Validate dims
        if table.dims().len() != 2 {
            return Err(MetalError::InvalidShape("embedding table must be [vocab, d_model]".to_string()));
        }
        let vocab = table.dims()[0];
        let d_model = table.dims()[1];

        let ind_dims = indices.dims();
        if ind_dims.is_empty() || ind_dims.len() > 2 {
            return Err(MetalError::InvalidShape("indices must be [n] or [batch, seq]".to_string()));
        }
        let (batch, seq) = if ind_dims.len() == 1 {
            (1usize, ind_dims[0])
        } else {
            (ind_dims[0], ind_dims[1])
        };

        ctx.prepare_tensors_for_active_cmd(&[table])?;
        indices.flush_host_writes()?;
        // Handle dependency checking for indices tensor manually
        if let Some(dep) = indices.defining_cmd_buffer.borrow().clone() {
            if let Some(active_cmd_buffer) = ctx.active_cmd_buffer.as_ref() {
                if !dep.ptr_eq(active_cmd_buffer) && !dep.is_completed() {
                    let completions = ctx.wait_for_command_buffer(dep.clone(), None);
                    ctx.process_pipeline_completions(completions);
                }
            } else if !dep.is_completed() {
                let completions = ctx.wait_for_command_buffer(dep.clone(), None);
                ctx.process_pipeline_completions(completions);
            }
            indices.defining_cmd_buffer.borrow_mut().take();
        }

        let out_dims = vec![batch, seq, d_model];
        let out = Tensor::new(out_dims.clone(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("embedding_lookup_op"));

        let pipeline = pipeline.expect("Kernel Library supplied for MetalKernels");

        let op = EmbeddingLookup::<T> {
            table: table.clone(),
            indices: indices.clone(),
            out: out.clone(),
            d_model: u32::try_from(d_model).map_err(|_| MetalError::InvalidShape("d_model exceeds u32".to_string()))?,
            total: u32::try_from(batch * seq * d_model).map_err(|_| MetalError::InvalidShape("output size exceeds u32".to_string()))?,
            vocab: u32::try_from(vocab).map_err(|_| MetalError::InvalidShape("vocab exceeds u32".to_string()))?,
            pipeline,
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for EmbeddingLookup<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let threads = 256usize;
        let total = self.total as usize;
        let groups = total.div_ceil(threads);

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(
                objc2_metal::MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.table.buf, self.table.offset);
        set_buffer(encoder, 1, &self.indices.buf, self.indices.offset);
        set_buffer(encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(encoder, 3, &self.d_model);
        set_bytes(encoder, 4, &self.total);
        set_bytes(encoder, 5, &self.vocab);
    }
}
