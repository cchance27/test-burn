use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use crate::{
    Context, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache, kernels::{
        DefaultKernelInvocable, matmul_dispatcher::{
            dispatcher::select_policy, execute::{MatmulDispatchArgs, execute}, types::{MatShape, MatmulCaps}
        }
    }
};

pub struct MatmulDispatchOp;

impl DefaultKernelInvocable for MatmulDispatchOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        Option<&'a Tensor<T>>, // bias
        Option<&'a Tensor<T>>, // out
        bool,                  // transpose_left
        bool,                  // transpose_right
        f32,                   // alpha
        f32,                   // beta
    );

    fn function_id() -> Option<crate::kernels::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        ctx.with_gpu_scope("matmul_dispatch", |ctx| {
            let (left, right, bias, out, transpose_left, transpose_right, alpha, beta) = args;
            let dtype = T::DTYPE; // dtype reserved for future selection; not strictly needed for now
            let left_dims = if transpose_left {
                [left.dims()[1], left.dims()[0]]
            } else {
                [left.dims()[0], left.dims()[1]]
            };
            let right_dims = if transpose_right {
                [right.dims()[1], right.dims()[0]]
            } else {
                [right.dims()[0], right.dims()[1]]
            };

            // For matmul C = A @ B:
            // - A has shape (m, k)
            // - B has shape (k, n)
            // - C has shape (m, n)
            let shape = MatShape {
                m: left_dims[0],  // rows of A
                k: left_dims[1],  // cols of A = rows of B
                n: right_dims[1], // cols of B
            };
            let caps = MatmulCaps {
                has_simdgroup_mm: ctx.device_has_simdgroup_mm(),
                max_tg_size: ctx.max_threads_per_threadgroup(),
            };
            let prefs = crate::kernels::matmul_dispatcher::prefs::load_prefs_from_env();

            // Environment-gated NOOP path for dispatcher overhead measurement.
            // Triggered by setting FORCE_MATMUL_BACKEND=noop.
            //let force_noop = metallic_env::FORCE_MATMUL_BACKEND
            //    .get()
            //    .ok()
            //    .flatten()
            //    .map(|s| s.eq_ignore_ascii_case("noop"))
            //    .unwrap_or(false);
            //if force_noop {
            //    let function_id = crate::kernels::KernelFunction::Noop;
            //    let pipeline = ctx.kernel_manager.get_pipeline(function_id, ctx.tensor_dtype(), &ctx.device)?;
            //    // Reuse provided out if present, otherwise allocate appropriately sized output.
            //    let out_tensor = if let Some(existing) = out {
            //        existing.clone()
            //    } else {
            //        Tensor::new(vec![shape.m, shape.n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?
            //    };
            //    return crate::kernels::tensors::NoopOp::new::<T>(ctx, out_tensor, Some(pipeline), cache);
            //}

            let plan = select_policy(shape, dtype, &caps, &prefs);
            let exec_args = MatmulDispatchArgs {
                left,
                right,
                bias,
                out,
                transpose_left,
                transpose_right,
                alpha,
                beta,
                dtype,
            };
            execute(ctx, plan, exec_args, cache)
        })
    }
}
