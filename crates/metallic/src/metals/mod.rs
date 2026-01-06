pub mod elemwise;
pub mod embedding;
pub mod gemv;
pub mod kv_cache_write;
pub mod kv_rearrange;
pub mod matmul_gemv;
pub mod repeat_kv_heads;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod sdpa;
pub mod softmax;
pub mod swiglu;
pub mod tensor;
pub mod v2;

// Re-exports for backwards compatibility (items moved to matmul_gemv/fused/)
pub use matmul_gemv::fused::{
    Q2FusedParams, QkvF16CanonicalFusedRmsnormArgs, QkvF16CanonicalFusedRmsnormStep, QkvFusedParams, SwiGluF16CanonicalFusedRmsnormArgs, SwiGluF16CanonicalFusedRmsnormStep, dispatch_qkv, dispatch_swiglu
};
