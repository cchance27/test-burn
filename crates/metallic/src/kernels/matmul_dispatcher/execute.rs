use super::types::*;
use crate::{
    Context, MetalError, Operation, Tensor, TensorElement, kernels::{
        DefaultKernelInvocable, KernelFunction, matmul_gemm_tiled::MatmulGemmTiledOp, matmul_gemv::MatmulGemvOp, matmul_gemv_smalln::{MatmulGemvSmallN1Op, MatmulGemvSmallN2Op, MatmulGemvSmallN4Op, MatmulGemvSmallN8Op, MatmulGemvSmallN16Op}, matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulMpsAlphaBetaOp, MatMulMpsOp}
    }, resource_cache::ResourceCache, tensor::Dtype
};

#[derive(Clone, Copy)]
pub struct MatmulDispatchArgs<'a, T: TensorElement> {
    pub left: &'a Tensor<T>,
    pub right: &'a Tensor<T>,
    pub bias: Option<&'a Tensor<T>>, // for MLX epilogue
    pub out: Option<&'a Tensor<T>>,  // optional destination (beta != 0 requires this)
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub alpha: f32,
    pub beta: f32,
    pub dtype: Dtype,
}

pub fn execute<'a, T: TensorElement>(
    ctx: &mut Context<T>,
    plan: DispatchPlan,
    args: MatmulDispatchArgs<'a, T>,
    mut cache: Option<&mut ResourceCache>,
) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
    let scope_label = format!("matmul_{}", plan);

    ctx.with_gpu_scope(scope_label, |ctx| {
        match plan {
            DispatchPlan::UseMLX(_variant) => {
                // Route through MLX GEMM with fused epilogue when possible
                let mlx_args = (
                    args.left,
                    args.right,
                    args.bias,
                    args.out,
                    args.transpose_left,
                    args.transpose_right,
                    args.alpha,
                    args.beta,
                );
                MatMulMlxOp::new(ctx, mlx_args, None, cache.as_deref_mut())
            }
            DispatchPlan::UseMPS(_variant) => {
                // If alpha==1 and beta==0, use fast MPS GEMM, else use alpha/beta op
                if args.alpha == 1.0 && args.beta == 0.0 {
                    let mps_args = (args.left, args.right, args.transpose_left, args.transpose_right);
                    MatMulMpsOp::new(ctx, mps_args, None, cache.as_deref_mut())
                } else {
                    // MPS alpha/beta path requires an existing output tensor as C
                    // If none provided and beta==0, we can allocate zeroed output and treat it as C
                    let out_tensor = if let Some(o) = args.out {
                        o.clone()
                    } else {
                        // Create output dims by introspecting left/right and transposes via MPS view
                        // Reuse MPS mod logic: create a temporary out by calling the base MPS op first
                        // Simpler: call MatMulMpsOp to get an output, then use it as C (it will be overwritten accordingly)
                        let tmp = MatMulMpsOp::new(
                            ctx,
                            (args.left, args.right, args.transpose_left, args.transpose_right),
                            None,
                            cache.as_deref_mut(),
                        )?;
                        tmp.1
                    };
                    let mps_ab_args = (
                        args.left,
                        args.right,
                        &out_tensor,
                        args.transpose_left,
                        args.transpose_right,
                        args.alpha,
                        args.beta,
                    );
                    MatMulMpsAlphaBetaOp::new(ctx, mps_ab_args, None, cache.as_deref_mut())
                }
            }
            DispatchPlan::Gemv(variant) => {
                // Legacy GEMV path currently only handles non-transposed inputs.
                if args.transpose_left || args.transpose_right {
                    return Err(MetalError::InvalidOperation(
                        "GEMV dispatch does not support transpose flags".to_string(),
                    ));
                }

                let left_dims = args.left.dims();
                let right_dims = args.right.dims();
                if left_dims.len() != 2 || right_dims.len() != 2 {
                    return Err(MetalError::InvalidShape(format!(
                        "GEMV dispatch expects rank-2 tensors, got left {:?}, right {:?}",
                        left_dims, right_dims
                    )));
                }

                if left_dims[1] != right_dims[0] {
                    return Err(MetalError::InvalidShape(format!(
                        "Incompatible shapes for GEMV dispatch: left {:?}, right {:?}",
                        left_dims, right_dims
                    )));
                }

                let is_vector_shape = left_dims[0] == 1;
                let gemm_args = (
                    args.left,
                    args.right,
                    args.bias,
                    args.out,
                    args.transpose_left,
                    args.transpose_right,
                    args.alpha,
                    args.beta,
                );

                match variant {
                    MatmulVariant::SmallN(SmallNBucket::N8) => {
                        if args.dtype == crate::Dtype::F16 {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemvSmallN8, args.dtype, &ctx.device)?;
                            MatmulGemvSmallN8Op::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                        }
                    }
                    MatmulVariant::SmallN(SmallNBucket::N1) => {
                        if args.dtype == crate::Dtype::F16 {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemvSmallN1, args.dtype, &ctx.device)?;
                            MatmulGemvSmallN1Op::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                        }
                    }
                    MatmulVariant::SmallN(SmallNBucket::N2) => {
                        if args.dtype == crate::Dtype::F16 {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemvSmallN2, args.dtype, &ctx.device)?;
                            MatmulGemvSmallN2Op::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                        }
                    }
                    MatmulVariant::SmallN(SmallNBucket::N4) => {
                        if args.dtype == crate::Dtype::F16 {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemvSmallN4, args.dtype, &ctx.device)?;
                            MatmulGemvSmallN4Op::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                        }
                    }
                    MatmulVariant::SmallN(SmallNBucket::N16) => {
                        if args.dtype == crate::Dtype::F16 {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemvSmallN16, args.dtype, &ctx.device)?;
                            MatmulGemvSmallN16Op::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                        }
                    }
                    MatmulVariant::GemmTiled(_) => {
                        // Use the new tiled GEMM kernel
                        MatmulGemmTiledOp::new(ctx, gemm_args, None, cache.as_deref_mut())
                    }
                    MatmulVariant::SmallN(_) | MatmulVariant::GemmSimd(_) | MatmulVariant::GemmGeneric => {
                        if is_vector_shape {
                            let pipeline = ctx
                                .kernel_manager
                                .get_pipeline(KernelFunction::MatmulGemv, args.dtype, &ctx.device)?;
                            MatmulGemvOp::new(ctx, (args.left, args.right), Some(pipeline), cache.as_deref_mut())
                        } else {
                            MatMulMlxOp::new(ctx, gemm_args, None, cache)
                        }
                    }
                }
            }
        }
    })
}
