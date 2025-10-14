use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use crate::{
    context::Context,
    error::MetalError,
    kernels::{
        softmax::SoftmaxOp,
        KernelInvocable, Operation,
    },
    resource_cache::ResourceCache,
    tensor::{Tensor, TensorElement, TensorInit, TensorStorage},
};

use super::{
    dispatcher::{select_policy, SoftmaxCaps, SoftmaxPrefs},
    execute::SoftmaxDispatch,
    types::{SoftmaxBackend, SoftmaxShape},
};

/// A public, zero-sized struct that acts as the entry point for the softmax dispatcher.
///
/// This can be called via `ctx.call::<SoftmaxDispatchOp>(...)`.
pub struct SoftmaxDispatchOp;

impl KernelInvocable for SoftmaxDispatchOp {
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

        // Create the destination tensor.
        let dst = Tensor::new(
            src.dims().to_vec(),
            TensorStorage::Pooled(ctx),
            TensorInit::Uninitialized,
        )?;

        // Determine the policy for this softmax operation.
        let shape = SoftmaxShape {
            seq_k: *src.dims().last().unwrap_or(&0),
        };
        // TODO: Get real caps from context and prefs from metallic_env.
        let caps = SoftmaxCaps::default();
        let prefs = SoftmaxPrefs::default();
        let policy = select_policy(shape, &caps, &prefs);

        let dims = src.dims();
        if dims.len() < 2 {
            return Err(MetalError::InvalidShape(
                "Softmax requires at least a 2D tensor".to_string(),
            ));
        }
        // Assuming 3D tensor [batch, seq_q, seq_k], but can handle 2D [seq_q, seq_k]
        let seq_k = dims[dims.len() - 1];
        let seq_q = dims[dims.len() - 2];

        // Based on the policy, create the appropriate underlying operation.
        let (op, out) = match policy.backend {
            SoftmaxBackend::Custom => {
                // For the custom kernel, we need to fetch its pipeline.
                let function_id =
                    SoftmaxOp::function_id().expect("SoftmaxOp should have a function_id");
                let pipeline = ctx
                    .kernel_manager
                    .get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
                SoftmaxOp::new(
                    ctx,
                    (
                        src,
                        seq_q as u32,
                        seq_q as u32,
                        seq_k as u32,
                        causal as u32,
                        query_offset,
                    ),
                    Some(pipeline),
                    cache,
                )?
            }
            SoftmaxBackend::MPS => {
                // MPS operations don't have a pre-compiled pipeline.
                //SoftmaxMpsOp::new(ctx, (src, &dst, causal), None, cache)?
                unreachable!("I don't think we've properly exposed the mps version ?");
            }
            SoftmaxBackend::Auto => {
                // This should have been resolved in select_policy.
                // We can fall back to a default or return an error.
                // For now, let's fall back to Custom.
                let function_id =
                    SoftmaxOp::function_id().expect("SoftmaxOp should have a function_id");
                let pipeline = ctx
                    .kernel_manager
                    .get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
                SoftmaxOp::new(
                    ctx,
                    (
                        src,
                        seq_q as u32,
                        seq_q as u32,
                        seq_k as u32,
                        causal as u32,
                        query_offset,
                    ),
                    Some(pipeline),
                    cache,
                )?
            }
        };

        // Wrap the underlying operation in our dispatch operation.
        let dispatch_op = SoftmaxDispatch::new(policy, op);

        Ok((Box::new(dispatch_op), out))
    }
}
