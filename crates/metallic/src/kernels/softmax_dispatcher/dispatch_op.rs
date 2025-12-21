use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use super::{
    dispatcher::{SoftmaxCaps, SoftmaxPrefs, select_policy}, execute::SoftmaxDispatch, prefs, types::{SoftmaxShape, SoftmaxVariant}
};
use crate::{
    caching::ResourceCache, context::Context, error::MetalError, kernels::{
        DefaultKernelInvocable, Operation, softmax_block::SoftmaxBlockOp, softmax_kernel::SoftmaxKernelOp, softmax_vec::SoftmaxVecOp
    }, tensor::{Dtype, Tensor, TensorElement}
};

/// A public, zero-sized struct that acts as the entry point for the softmax dispatcher.
///
/// This can be called via `ctx.call::<SoftmaxDispatchOp>(..., None)`.
pub struct SoftmaxDispatchOp;

impl DefaultKernelInvocable for SoftmaxDispatchOp {
    /// (input_tensor, causal_mask_flag, query_offset)
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, bool, u32);

    /// This is a virtual op that dispatches to other ops, so it doesn't have its own kernel function.
    fn function_id() -> Option<crate::kernels::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (src, causal, query_offset) = args;

        // Environment-gated NOOP path for measuring dispatcher overhead.
        // Triggered by setting METALLIC_SOFTMAX_BACKEND=noop or METALLIC_SOFTMAX_VARIANT=noop.
        let force_noop = std::env::var("METALLIC_SOFTMAX_BACKEND")
            .map(|s| s.eq_ignore_ascii_case("noop"))
            .unwrap_or(false)
            || std::env::var("METALLIC_SOFTMAX_VARIANT")
                .map(|s| s.eq_ignore_ascii_case("noop"))
                .unwrap_or(false);
        if force_noop {
            let function_id = crate::kernels::KernelFunction::Noop;
            let pipeline = ctx.kernel_manager.get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
            // Return the source tensor unchanged; this mimics in-place softmax result shape.
            return crate::kernels::tensors::NoopOp::new(ctx, src.clone(), Some(pipeline), cache);
        }

        // Determine the policy for this softmax operation.
        let shape = SoftmaxShape {
            seq_k: *src.dims().last().unwrap_or(&0),
        };
        // Load caps and preferences (may be forced via environment variables)
        let caps = SoftmaxCaps::default();
        let loaded = prefs::load_prefs_from_env();
        let prefs = SoftmaxPrefs {
            forced_variant: loaded.forced_variant,
            forced_tg_size: loaded.forced_tg_size,
        };
        let policy = select_policy(shape, &caps, &prefs);

        let dims = src.dims();
        if dims.len() < 2 {
            return Err(MetalError::InvalidShape("Softmax requires at least a 2D tensor".to_string()));
        }
        // Calculate dimensions the same way as the original softmax function
        // Assuming 3D tensor [batch, seq_q, seq_k], but can handle 2D [seq_q, seq_k]
        let seq_k = dims[dims.len() - 1];
        let seq_q = dims[dims.len() - 2];
        let batch = if dims.len() > 2 { dims[0] } else { 1 }; // Handle 2D case

        // Calculate rows_total the same way as original softmax: batch * seq_q
        let rows_total = (batch * seq_q) as u32;

        // Take the pending GPU scope to consume it, as required by the kernel infrastructure
        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| crate::context::GpuProfilerLabel::fallback("softmax_dispatch_op"));

        // Based on the policy, create the appropriate underlying operation.
        let (op, out) = match policy.variant {
            SoftmaxVariant::Vec if matches!(ctx.tensor_dtype(), Dtype::F16 | Dtype::F32) => {
                let function_id = SoftmaxVecOp::function_id().expect("SoftmaxVecOp should have a function_id");
                let pipeline = ctx.kernel_manager.get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
                SoftmaxVecOp::new(
                    ctx,
                    (src, rows_total, seq_q as u32, seq_k as u32, causal as u32, query_offset),
                    Some(pipeline),
                    cache,
                )?
            }
            SoftmaxVariant::Block if matches!(ctx.tensor_dtype(), Dtype::F16 | Dtype::F32) => {
                let function_id = SoftmaxBlockOp::function_id().expect("SoftmaxBlockOp should have a function_id");
                let pipeline = ctx.kernel_manager.get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
                SoftmaxBlockOp::new(
                    ctx,
                    (src, rows_total, seq_q as u32, seq_k as u32, 1024u32, causal as u32, query_offset),
                    Some(pipeline),
                    cache,
                )?
            }
            SoftmaxVariant::Vec | SoftmaxVariant::Block | SoftmaxVariant::Auto => {
                let function_id = SoftmaxKernelOp::function_id().expect("SoftmaxKernelOp should have a function_id");
                let pipeline = ctx.kernel_manager.get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
                SoftmaxKernelOp::new(
                    ctx,
                    (src, rows_total, seq_q as u32, seq_k as u32, causal as u32, query_offset),
                    Some(pipeline),
                    cache,
                )?
            }
        };

        // Wrap the underlying operation in our dispatch operation.
        let dispatch_op = SoftmaxDispatch::new_with_label(policy, op, profiler_label);

        Ok((Box::new(dispatch_op), out))
    }
}
