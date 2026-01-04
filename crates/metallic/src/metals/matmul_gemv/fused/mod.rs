//! Fused GEMV operations for Foundry.
//!
//! This module contains fused kernels that combine normalization + GEMV + activation
//! into single GPU dispatches for decode-time performance.

pub mod qkv;
pub mod swiglu;

pub use qkv::{
    QkvF16CanonicalFusedRmsnormArgs, QkvF16CanonicalFusedRmsnormStep, QkvFusedParams, dispatch_qkv, qkv_f16_canonical_fused_rmsnorm_kernel
};
pub use swiglu::{
    Q2FusedParams, SwiGluF16CanonicalFusedRmsnormArgs, SwiGluF16CanonicalFusedRmsnormStep, dispatch_swiglu, swiglu_f16_canonical_fused_rmsnorm_kernel
};
