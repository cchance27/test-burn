use crate::{Context, MetalError, Operation, Tensor, TensorElement};
use crate::kernels::{KernelInvocable};
use crate::resource_cache::ResourceCache;
use crate::tensor::Dtype;
use super::types::*;
use crate::kernels::{matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulMpsOp, MatMulMpsAlphaBetaOp}, matmul_gemv::MatmulGemvOp};

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
                    let out_tensor = if let Some(o) = args.out { o.clone() } else {
                        // Create output dims by introspecting left/right and transposes via MPS view
                        // Reuse MPS mod logic: create a temporary out by calling the base MPS op first
                        // Simpler: call MatMulMpsOp to get an output, then use it as C (it will be overwritten accordingly)
                        let tmp = MatMulMpsOp::new(ctx, (args.left, args.right, args.transpose_left, args.transpose_right), None, cache.as_deref_mut())?;
                        tmp.1
                    };
                    let mps_ab_args = (args.left, args.right, &out_tensor, args.transpose_left, args.transpose_right, args.alpha, args.beta);
                    MatMulMpsAlphaBetaOp::new(ctx, mps_ab_args, None, cache.as_deref_mut())
                }
            }
            DispatchPlan::UseLegacyGemv(_variant) => {
                // Legacy GEMV expects (x, A) with x as (1, K) and A as (K, N)
                // We only support transpose_right=false here; more variants may be added later
                if args.transpose_left || args.transpose_right {
                    return Err(MetalError::InvalidOperation("Legacy GEMV does not support transpose flags".to_string()));
                }
                MatmulGemvOp::new(ctx, (args.left, args.right), None, cache)
            }
        }
    })
}
