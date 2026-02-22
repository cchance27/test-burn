use std::sync::Arc;

use half::f16;

use crate::{
    F16, Foundry, MetalError, compound::Layout, policy::{MetalPolicyRuntime, activation::Activation}, spec::{FastBindings, ResolvedSymbols}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit}, types::{TensorArg, dispatch::DispatchConfig}
};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_canonical_projection(
    foundry: &mut Foundry,
    fast_bindings: &FastBindings,
    policy: Arc<dyn MetalPolicyRuntime>,
    resolved: &ResolvedSymbols,
    normalized: &TensorArg,
    dst: TensorArg,
    k_dim: u32,
    n_dim: u32,
    batch: u32,
    fallback_weights_per_block: u32,
) -> Result<(), MetalError> {
    let loader = policy.loader_stage();
    let bound = loader.bind(fast_bindings, resolved);
    let weights = bound[0].clone();
    let scale_bytes = if bound.len() > 1 { bound[1].clone() } else { weights.clone() };
    let weights_per_block = if policy.has_scale() {
        policy.meta().weights_per_block as u32
    } else {
        fallback_weights_per_block
    };

    let kernel = crate::metals::gemv::step::get_gemv_v2_kernel(
        policy,
        Layout::Canonical {
            expected_k: k_dim as usize,
            expected_n: n_dim as usize,
        },
        crate::metals::gemv::step::GemvStrategy::Canonical,
        Activation::None,
    );
    let args = crate::metals::gemv::step::GemvV2Args {
        weights,
        scale_bytes,
        input: normalized.clone(),
        output: dst.clone(),
        k_dim,
        n_dim,
        weights_per_block,
        bias: dst.clone(),
        has_bias: 0,
        alpha: 1.0,
        residual: dst,
        has_residual: 0,
        beta: 0.0,
    };
    let dispatch = DispatchConfig::warp_per_row(n_dim, batch);
    foundry.run(&kernel.bind_arc(args, dispatch))
}

pub(super) fn allocate_zero_bias(foundry: &mut Foundry, n_dim: u32) -> Result<FoundryTensor<F16, Pooled>, MetalError> {
    let zero_bias = vec![f16::from_f32(0.0); n_dim as usize];
    FoundryTensor::<F16, Pooled>::new(foundry, vec![n_dim as usize], TensorInit::CopyFrom(&zero_bias))
        .map_err(|e| MetalError::OperationFailed(format!("swiglu zero bias alloc failed: {e}")))
}
