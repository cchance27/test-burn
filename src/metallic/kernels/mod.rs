use crate::metallic::{
    Context, Dtype, MetalError, Operation, Tensor,
    encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state},
    resource_cache::ResourceCache,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder as _, MTLComputePipelineState, MTLSize};
use objc2_metal::{MTLDevice, MTLLibrary};
use rustc_hash::FxHashMap;

// Rexport KernelManager and Invocable
mod kernel_manager;
pub use kernel_manager::{KernelInvocable, KernelManager};

// Export our kernels
pub mod elemwise_add;
pub mod elemwise_div;
pub mod elemwise_mul;
pub mod elemwise_sub;
pub mod gelu;
pub mod gemv;
pub mod kv_cache_write;
pub mod kv_rearrange;
pub mod layernorm;
pub mod matmul;
pub mod mlxmatmul;
pub mod permute;
pub mod repeat_kv_heads;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod scaled_dot_product_attention;
pub mod silu;
pub mod softmax;
pub mod swiglu;
pub mod tensors;

/// Uniquely identifies a compiled Metal library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelLibrary {
    Cast,
    ElemwiseAdd,
    ElemwiseDiv,
    ElemwiseMul,
    ElemwiseSub,
    Gelu,
    KvCacheWrite,
    KvRearrange,
    LayerNorm,
    Permute,
    RepeatKvHeads,
    Rope,
    RMSNorm,
    Silu,
    Sampling,
    Softmax,
    Swiglu,
    Tensors,
    Gemv,
}

impl KernelLibrary {
    fn source(&self) -> &'static str {
        match self {
            KernelLibrary::Cast => include_str!("cast/kernel.metal"),
            KernelLibrary::ElemwiseAdd => include_str!("elemwise_add/kernel.metal"),
            KernelLibrary::ElemwiseDiv => include_str!("elemwise_div/kernel.metal"),
            KernelLibrary::ElemwiseMul => include_str!("elemwise_mul/kernel.metal"),
            KernelLibrary::ElemwiseSub => include_str!("elemwise_sub/kernel.metal"),
            KernelLibrary::Gelu => include_str!("gelu/kernel.metal"),
            KernelLibrary::KvCacheWrite => include_str!("kv_cache_write/kernel.metal"),
            KernelLibrary::KvRearrange => include_str!("kv_rearrange/kernel.metal"),
            KernelLibrary::LayerNorm => include_str!("layernorm/kernel.metal"),
            KernelLibrary::Permute => include_str!("permute/kernel.metal"),
            KernelLibrary::RepeatKvHeads => include_str!("repeat_kv_heads/kernel.metal"),
            KernelLibrary::Rope => include_str!("rope/kernel.metal"),
            KernelLibrary::RMSNorm => include_str!("rmsnorm/kernel.metal"),
            KernelLibrary::Silu => include_str!("silu/kernel.metal"),
            KernelLibrary::Sampling => include_str!("sampling/kernel.metal"),
            KernelLibrary::Softmax => include_str!("softmax/kernel.metal"),
            KernelLibrary::Swiglu => include_str!("swiglu/kernel.metal"),
            KernelLibrary::Tensors => include_str!("tensors/kernel.metal"),
            KernelLibrary::Gemv => include_str!("gemv/kernel.metal"),
        }
    }
}

/// Uniquely identifies a function within a Metal library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelFunction {
    CastToF16,
    CastFromF16,
    CastToF32,
    CastFromF32,
    ElemwiseAdd,
    ElemwiseBroadcastAdd,
    ElemwiseDiv,
    ElemwiseMul,
    ElemwiseSub,
    Gelu,
    KvCacheWrite,
    KvRearrange,
    LayerNorm,
    Permute,
    RepeatKvHeads,
    Rope,
    RMSNorm,
    Silu,
    FusedSoftmax,
    SwigluFusedActivation,
    Arange,
    Ones,
    RandomUniform,
    Gemv,
    SamplerF32,
    SamplerF16,
}

impl KernelFunction {
    fn library(&self) -> KernelLibrary {
        match self {
            KernelFunction::CastToF16 | KernelFunction::CastFromF16 | KernelFunction::CastToF32 | KernelFunction::CastFromF32 => {
                KernelLibrary::Cast
            }
            KernelFunction::ElemwiseAdd | KernelFunction::ElemwiseBroadcastAdd => KernelLibrary::ElemwiseAdd,
            KernelFunction::ElemwiseDiv => KernelLibrary::ElemwiseDiv,
            KernelFunction::ElemwiseMul => KernelLibrary::ElemwiseMul,
            KernelFunction::ElemwiseSub => KernelLibrary::ElemwiseSub,
            KernelFunction::Gelu => KernelLibrary::Gelu,
            KernelFunction::KvCacheWrite => KernelLibrary::KvCacheWrite,
            KernelFunction::KvRearrange => KernelLibrary::KvRearrange,
            KernelFunction::LayerNorm => KernelLibrary::LayerNorm,
            KernelFunction::Permute => KernelLibrary::Permute,
            KernelFunction::RepeatKvHeads => KernelLibrary::RepeatKvHeads,
            KernelFunction::Rope => KernelLibrary::Rope,
            KernelFunction::RMSNorm => KernelLibrary::RMSNorm,
            KernelFunction::Silu => KernelLibrary::Silu,
            KernelFunction::FusedSoftmax => KernelLibrary::Softmax,
            KernelFunction::SwigluFusedActivation => KernelLibrary::Swiglu,
            KernelFunction::Arange | KernelFunction::Ones | KernelFunction::RandomUniform => KernelLibrary::Tensors,
            KernelFunction::Gemv => KernelLibrary::Gemv,
            KernelFunction::SamplerF32 | KernelFunction::SamplerF16 => KernelLibrary::Sampling,
        }
    }

    fn name_for_dtype(&self, dtype: Dtype) -> Result<&'static str, MetalError> {
        use Dtype::*;

        let name = match (self, dtype) {
            (KernelFunction::CastToF16, F32) => "cast_to_f16_kernel_f32",
            (KernelFunction::CastToF16, F16) => "cast_to_f16_kernel_f16",
            (KernelFunction::CastFromF16, F32) => "cast_from_f16_kernel_f32",
            (KernelFunction::CastFromF16, F16) => "cast_from_f16_kernel_f16",
            (KernelFunction::CastToF32, F32) => "cast_to_f32_kernel_f32",
            (KernelFunction::CastToF32, F16) => "cast_to_f32_kernel_f16",
            (KernelFunction::CastFromF32, F32) => "cast_from_f32_kernel_f32",
            (KernelFunction::CastFromF32, F16) => "cast_from_f32_kernel_f16",
            (KernelFunction::ElemwiseAdd, F32) => "add_kernel_f32",
            (KernelFunction::ElemwiseAdd, F16) => "add_kernel_f16",
            (KernelFunction::ElemwiseBroadcastAdd, F32) => "broadcast_add_kernel_f32",
            (KernelFunction::ElemwiseBroadcastAdd, F16) => "broadcast_add_kernel_f16",
            (KernelFunction::ElemwiseDiv, F32) => "div_kernel_f32",
            (KernelFunction::ElemwiseDiv, F16) => "div_kernel_f16",
            (KernelFunction::ElemwiseMul, F32) => "mul_kernel_f32",
            (KernelFunction::ElemwiseMul, F16) => "mul_kernel_f16",
            (KernelFunction::ElemwiseSub, F32) => "sub_kernel_f32",
            (KernelFunction::ElemwiseSub, F16) => "sub_kernel_f16",
            (KernelFunction::Gelu, F32) => "gelu_kernel_f32",
            (KernelFunction::Gelu, F16) => "gelu_kernel_f16",
            (KernelFunction::KvCacheWrite, F32) => "kv_cache_write_kernel_f32",
            (KernelFunction::KvCacheWrite, F16) => "kv_cache_write_kernel_f16",
            (KernelFunction::KvRearrange, F32) => "kv_rearrange_kernel_f32",
            (KernelFunction::KvRearrange, F16) => "kv_rearrange_kernel_f16",
            (KernelFunction::LayerNorm, F32) => "layernorm_kernel_f32",
            (KernelFunction::LayerNorm, F16) => "layernorm_kernel_f16",
            (KernelFunction::Permute, F32) => "permute_kernel_f32",
            (KernelFunction::Permute, F16) => "permute_kernel_f16",
            (KernelFunction::RepeatKvHeads, F32) => "repeat_kv_heads_kernel_f32",
            (KernelFunction::RepeatKvHeads, F16) => "repeat_kv_heads_kernel_f16",
            (KernelFunction::Rope, F32) => "rope_kernel_f32",
            (KernelFunction::Rope, F16) => "rope_kernel_f16",
            (KernelFunction::RMSNorm, F32) => "rmsnorm_kernel_f32",
            (KernelFunction::RMSNorm, F16) => "rmsnorm_kernel_f16",
            (KernelFunction::Silu, F32) => "silu_kernel_f32",
            (KernelFunction::Silu, F16) => "silu_kernel_f16",
            (KernelFunction::SwigluFusedActivation, F32) => "swiglu_fused_activation_f32",
            (KernelFunction::SwigluFusedActivation, F16) => "swiglu_fused_activation_f16",
            (KernelFunction::FusedSoftmax, F32) => "sdpa_fused_softmax_f32",
            (KernelFunction::FusedSoftmax, F16) => "sdpa_fused_softmax_f16",
            (KernelFunction::Arange, F32) => "arange_kernel_f32",
            (KernelFunction::Arange, F16) => "arange_kernel_f16",
            (KernelFunction::Ones, F32) => "ones_kernel_f32",
            (KernelFunction::Ones, F16) => "ones_kernel_f16",
            (KernelFunction::RandomUniform, F32) => "random_uniform_f32",
            (KernelFunction::RandomUniform, F16) => "random_uniform_f16",
            (KernelFunction::Gemv, F32) => "gemv_f32",
            (KernelFunction::Gemv, F16) => "gemv_f16",
            (KernelFunction::SamplerF32, F32) => "sample_top_k_top_p_f32",
            (KernelFunction::SamplerF32, other) => {
                return Err(MetalError::UnsupportedDtype {
                    operation: "gpu_sampler",
                    dtype: other,
                });
            }
            (KernelFunction::SamplerF16, F16) => "sample_top_k_top_p_f16",
            (KernelFunction::SamplerF16, other) => {
                return Err(MetalError::UnsupportedDtype {
                    operation: "gpu_sampler",
                    dtype: other,
                });
            }
        };

        Ok(name)
    }
}
