use std::sync::Arc;

use crate::{
    compound::CompiledCompoundKernel, fusion::MetalPolicy, kernel_registry::{KernelCacheKey, kernel_registry}
};

#[inline]
pub fn get_or_build_compound_kernel<F>(family: impl Into<String>, variant: impl Into<String>, builder: F) -> Arc<CompiledCompoundKernel>
where
    F: FnOnce() -> CompiledCompoundKernel,
{
    let key = KernelCacheKey::new(family, variant);
    kernel_registry().get_or_build(key, builder)
}

#[inline]
pub fn get_or_build_policy_compound_kernel<F>(
    family: impl Into<String>,
    policy: Arc<dyn MetalPolicy>,
    builder: F,
) -> Arc<CompiledCompoundKernel>
where
    F: FnOnce(Arc<dyn MetalPolicy>) -> CompiledCompoundKernel,
{
    let variant = policy.short_name().to_string();
    get_or_build_compound_kernel(family, variant, move || builder(policy))
}
