use std::sync::Arc;

use smallvec::SmallVec;

use crate::{
    MetalError, fusion::MetalPolicy, policy::MetalPolicyRuntime, spec::{FastBindings, ResolvedSymbols}, tensor::Dtype, types::TensorArg
};

#[derive(Clone)]
pub struct BoundPolicyArgs {
    pub policy: Arc<dyn MetalPolicyRuntime>,
    pub args: SmallVec<[TensorArg; 4]>,
}

impl BoundPolicyArgs {
    #[inline]
    pub fn weight(&self) -> TensorArg {
        self.args[0].clone()
    }

    #[inline]
    pub fn scale(&self) -> Option<TensorArg> {
        self.args.get(1).cloned()
    }
}

#[derive(Clone)]
pub struct DualPolicyBindings {
    pub same_policy: bool,
    pub a: BoundPolicyArgs,
    pub b: BoundPolicyArgs,
}

#[derive(Clone)]
pub struct TriplePolicyBindings {
    pub same_policy: bool,
    pub a: BoundPolicyArgs,
    pub b: BoundPolicyArgs,
    pub c: BoundPolicyArgs,
}

#[inline]
fn bind_args(policy: &Arc<dyn MetalPolicyRuntime>, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> SmallVec<[TensorArg; 4]> {
    policy.loader_stage().bind(fast_bindings, resolved)
}

pub fn bind_dual_policy_slots(
    fast_bindings: &FastBindings,
    a_dtype: Dtype,
    a_resolved: &ResolvedSymbols,
    b_dtype: Dtype,
    b_resolved: &ResolvedSymbols,
) -> DualPolicyBindings {
    if a_dtype == b_dtype {
        let shared = crate::policy::resolve_policy(a_dtype);
        let loader = shared.loader_stage();
        return DualPolicyBindings {
            same_policy: true,
            a: BoundPolicyArgs {
                policy: shared.clone(),
                args: loader.bind(fast_bindings, a_resolved),
            },
            b: BoundPolicyArgs {
                policy: shared,
                args: loader.bind(fast_bindings, b_resolved),
            },
        };
    }

    let a_policy = crate::policy::resolve_policy(a_dtype);
    let b_policy = crate::policy::resolve_policy(b_dtype);
    DualPolicyBindings {
        same_policy: false,
        a: BoundPolicyArgs {
            policy: a_policy.clone(),
            args: bind_args(&a_policy, fast_bindings, a_resolved),
        },
        b: BoundPolicyArgs {
            policy: b_policy.clone(),
            args: bind_args(&b_policy, fast_bindings, b_resolved),
        },
    }
}

pub fn bind_triple_policy_slots(
    fast_bindings: &FastBindings,
    a_dtype: Dtype,
    a_resolved: &ResolvedSymbols,
    b_dtype: Dtype,
    b_resolved: &ResolvedSymbols,
    c_dtype: Dtype,
    c_resolved: &ResolvedSymbols,
) -> TriplePolicyBindings {
    if a_dtype == b_dtype && a_dtype == c_dtype {
        let shared = crate::policy::resolve_policy(a_dtype);
        let loader = shared.loader_stage();
        return TriplePolicyBindings {
            same_policy: true,
            a: BoundPolicyArgs {
                policy: shared.clone(),
                args: loader.bind(fast_bindings, a_resolved),
            },
            b: BoundPolicyArgs {
                policy: shared.clone(),
                args: loader.bind(fast_bindings, b_resolved),
            },
            c: BoundPolicyArgs {
                policy: shared,
                args: loader.bind(fast_bindings, c_resolved),
            },
        };
    }

    let a_policy = crate::policy::resolve_policy(a_dtype);
    let b_policy = if b_dtype == a_dtype {
        a_policy.clone()
    } else {
        crate::policy::resolve_policy(b_dtype)
    };
    let c_policy = if c_dtype == a_dtype {
        a_policy.clone()
    } else if c_dtype == b_dtype {
        b_policy.clone()
    } else {
        crate::policy::resolve_policy(c_dtype)
    };

    TriplePolicyBindings {
        same_policy: false,
        a: BoundPolicyArgs {
            policy: a_policy.clone(),
            args: bind_args(&a_policy, fast_bindings, a_resolved),
        },
        b: BoundPolicyArgs {
            policy: b_policy.clone(),
            args: bind_args(&b_policy, fast_bindings, b_resolved),
        },
        c: BoundPolicyArgs {
            policy: c_policy.clone(),
            args: bind_args(&c_policy, fast_bindings, c_resolved),
        },
    }
}

/// Ordered role→policy tuple key used for multi-policy kernel variant caching.
#[inline]
pub fn tuple_variant_key(slots: &[(&str, &dyn MetalPolicy)]) -> String {
    let mut cap = 0usize;
    for (role, policy) in slots {
        cap += role.len() + 1 + policy.short_name().len() + 1;
    }

    let mut key = String::with_capacity(cap.saturating_sub(1));
    for (idx, (role, policy)) in slots.iter().enumerate() {
        if idx != 0 {
            key.push(',');
        }
        key.push_str(role);
        key.push('=');
        key.push_str(policy.short_name());
    }
    key
}

/// Resolve the effective weights-per-block for a policy role.
///
/// Quantized policies are authoritative via metadata. Dense policies use a caller fallback
/// to preserve existing canonical fast-path behavior where relevant.
#[inline]
pub fn effective_weights_per_block(policy: &dyn MetalPolicy, fallback: u32) -> u32 {
    if policy.has_scale() {
        policy.meta().weights_per_block as u32
    } else {
        fallback
    }
}

/// Return a conservative vector width in elements for a tuple of policy roles.
///
/// Mapping mirrors existing single-policy behavior:
/// - `8` bytes → 8-lane path
/// - `16` bytes (F32-native float4 loads) → 4-lane math path
/// - `1/2/4` bytes → 4-lane path
pub fn tuple_vector_width_elements(slots: &[(&str, &dyn MetalPolicy)]) -> Result<u32, MetalError> {
    let mut width = 8u32;
    for (role, policy) in slots {
        let candidate = match policy.optimization_hints().vector_load_size {
            8 => 8,
            1 | 2 | 4 | 16 => 4,
            other => {
                return Err(MetalError::OperationNotSupported(format!(
                    "Unsupported vector_load_size={} for role '{}' (policy={})",
                    other,
                    role,
                    policy.short_name()
                )));
            }
        };
        width = width.min(candidate);
    }
    Ok(width.max(4))
}
