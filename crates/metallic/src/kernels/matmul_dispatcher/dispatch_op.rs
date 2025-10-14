use crate::kernels::matmul_dispatcher::{dispatcher::select_policy, execute::{execute, MatmulDispatchArgs}, types::{MatmulCaps, MatShape}};
use crate::kernels::KernelInvocable;
use crate::{Context, MetalError, Operation, Tensor, TensorElement};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
use crate::resource_cache::ResourceCache;

pub struct MatmulDispatchOp;

impl KernelInvocable for MatmulDispatchOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        Option<&'a Tensor<T>>, // bias
        Option<&'a Tensor<T>>, // out
        bool, // transpose_left
        bool, // transpose_right
        f32,  // alpha
        f32,  // beta
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
            let shape = MatShape { m: left.dims().last().copied().unwrap_or(0), k: left.dims().first().copied().unwrap_or(0), n: right.dims().last().copied().unwrap_or(0) };
            let caps = MatmulCaps { has_simdgroup_mm: ctx.device_has_simdgroup_mm(), max_tg_size: ctx.max_threads_per_threadgroup() };
            let prefs = crate::kernels::matmul_dispatcher::prefs::load_prefs_from_env();
            let plan = select_policy(shape, dtype, &caps, &prefs);
            let exec_args = MatmulDispatchArgs { left, right, bias, out, transpose_left, transpose_right, alpha, beta, dtype };
            execute(ctx, plan, exec_args, cache)
        })
    }
}
