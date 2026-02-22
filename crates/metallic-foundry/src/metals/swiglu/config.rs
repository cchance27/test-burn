use crate::{fusion::MetalPolicy, spec::TensorBindings};

#[inline]
pub(super) fn resolve_rms_eps(bindings: &TensorBindings, fallback: Option<f32>) -> f32 {
    bindings.get_f32_var("rms_eps").or(fallback).unwrap_or(1e-6)
}

#[inline]
pub(super) fn effective_weights_per_block<P: MetalPolicy + ?Sized>(policy: &P, fallback: u32) -> u32 {
    if policy.has_scale() {
        policy.meta().weights_per_block as u32
    } else {
        fallback
    }
}
