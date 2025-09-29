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
pub mod fused_qkv;
pub mod gelu;
pub mod kv_rearrange;
pub mod layernorm;
pub mod matmul;
pub mod permute;
pub mod repeat_kv_heads;
pub mod rmsnorm;
pub mod rope;
pub mod scaled_dot_product_attention;
pub mod silu;
pub mod softmax;
pub mod swiglu;
pub mod tensors;

/// Uniquely identifies a compiled Metal library.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum KernelLibrary {
    ElemwiseAdd,
    ElemwiseDiv,
    ElemwiseMul,
    ElemwiseSub,
    FusedQkv,
    Gelu,
    KvRearrange,
    LayerNorm,
    Permute,
    RepeatKvHeads,
    Rope,
    RMSNorm,
    Silu,
    Softmax,
    Tensors,
}

impl KernelLibrary {
    fn source(&self) -> &'static str {
        match self {
            KernelLibrary::ElemwiseAdd => include_str!("elemwise_add/kernel.metal"),
            KernelLibrary::ElemwiseDiv => include_str!("elemwise_div/kernel.metal"),
            KernelLibrary::ElemwiseMul => include_str!("elemwise_mul/kernel.metal"),
            KernelLibrary::ElemwiseSub => include_str!("elemwise_sub/kernel.metal"),
            KernelLibrary::FusedQkv => include_str!("fused_qkv/kernel.metal"),
            KernelLibrary::Gelu => include_str!("gelu/kernel.metal"),
            KernelLibrary::KvRearrange => include_str!("kv_rearrange/kernel.metal"),
            KernelLibrary::LayerNorm => include_str!("layernorm/kernel.metal"),
            KernelLibrary::Permute => include_str!("permute/kernel.metal"),
            KernelLibrary::RepeatKvHeads => include_str!("repeat_kv_heads/kernel.metal"),
            KernelLibrary::Rope => include_str!("rope/kernel.metal"),
            KernelLibrary::RMSNorm => include_str!("rmsnorm/kernel.metal"),
            KernelLibrary::Silu => include_str!("silu/kernel.metal"),
            KernelLibrary::Softmax => include_str!("softmax/kernel.metal"),
            KernelLibrary::Tensors => include_str!("tensors/kernel.metal"),
        }
    }
}

/// Uniquely identifies a function within a Metal library.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum KernelFunction {
    ElemwiseAdd,
    ElemwiseBroadcastAdd,
    ElemwiseDiv,
    ElemwiseMul,
    ElemwiseSub,
    FusedQkvBiasSplit,
    Gelu,
    KvRearrange,
    LayerNorm,
    Permute,
    RepeatKvHeads,
    Rope,
    RMSNorm,
    Silu,
    FusedSoftmax,
    Arange,
    Ones,
    RandomUniform,
}

impl KernelFunction {
    fn library(&self) -> KernelLibrary {
        match self {
            KernelFunction::ElemwiseAdd | KernelFunction::ElemwiseBroadcastAdd => KernelLibrary::ElemwiseAdd,
            KernelFunction::ElemwiseDiv => KernelLibrary::ElemwiseDiv,
            KernelFunction::ElemwiseMul => KernelLibrary::ElemwiseMul,
            KernelFunction::ElemwiseSub => KernelLibrary::ElemwiseSub,
            KernelFunction::FusedQkvBiasSplit => KernelLibrary::FusedQkv,
            KernelFunction::Gelu => KernelLibrary::Gelu,
            KernelFunction::KvRearrange => KernelLibrary::KvRearrange,
            KernelFunction::LayerNorm => KernelLibrary::LayerNorm,
            KernelFunction::Permute => KernelLibrary::Permute,
            KernelFunction::RepeatKvHeads => KernelLibrary::RepeatKvHeads,
            KernelFunction::Rope => KernelLibrary::Rope,
            KernelFunction::RMSNorm => KernelLibrary::RMSNorm,
            KernelFunction::Silu => KernelLibrary::Silu,
            KernelFunction::FusedSoftmax => KernelLibrary::Softmax,
            KernelFunction::Arange | KernelFunction::Ones | KernelFunction::RandomUniform => KernelLibrary::Tensors,
        }
    }

    fn name_for_dtype(&self, dtype: Dtype) -> &'static str {
        match self {
            KernelFunction::ElemwiseAdd => match dtype {
                Dtype::F32 => "add_kernel_f32",
                Dtype::F16 => "add_kernel_f16",
                Dtype::BF16 => "add_kernel_bf16",
                _ => "add_kernel_f32",
            },
            KernelFunction::ElemwiseBroadcastAdd => match dtype {
                Dtype::F32 => "broadcast_add_kernel_f32",
                Dtype::F16 => "broadcast_add_kernel_f16",
                Dtype::BF16 => "broadcast_add_kernel_bf16",
                _ => "broadcast_add_kernel_f32",
            },
            KernelFunction::ElemwiseDiv => match dtype {
                Dtype::F32 => "div_kernel_f32",
                Dtype::F16 => "div_kernel_f16",
                Dtype::BF16 => "div_kernel_bf16",
                _ => "div_kernel_f32",
            },
            KernelFunction::ElemwiseMul => match dtype {
                Dtype::F32 => "mul_kernel_f32",
                Dtype::F16 => "mul_kernel_f16",
                Dtype::BF16 => "mul_kernel_bf16",
                _ => "mul_kernel_f32",
            },
            KernelFunction::ElemwiseSub => match dtype {
                Dtype::F32 => "sub_kernel_f32",
                Dtype::F16 => "sub_kernel_f16",
                Dtype::BF16 => "sub_kernel_bf16",
                _ => "sub_kernel_f32",
            },
            KernelFunction::FusedQkvBiasSplit => match dtype {
                Dtype::F32 => "fused_qkv_bias_split_f32",
                Dtype::F16 => "fused_qkv_bias_split_f16",
                Dtype::BF16 => "fused_qkv_bias_split_bf16",
                _ => "fused_qkv_bias_split_f32",
            },
            KernelFunction::Gelu => match dtype {
                Dtype::F32 => "gelu_kernel_f32",
                Dtype::F16 => "gelu_kernel_f16",
                Dtype::BF16 => "gelu_kernel_bf16",
                _ => "gelu_kernel_f32",
            },
            KernelFunction::KvRearrange => match dtype {
                Dtype::F32 => "kv_rearrange_kernel_f32",
                Dtype::F16 => "kv_rearrange_kernel_f16",
                Dtype::BF16 => "kv_rearrange_kernel_bf16",
                _ => "kv_rearrange_kernel_f32",
            },
            KernelFunction::LayerNorm => match dtype {
                Dtype::F32 => "layernorm_kernel_f32",
                Dtype::F16 => "layernorm_kernel_f16",
                Dtype::BF16 => "layernorm_kernel_bf16",
                _ => "layernorm_kernel_f32",
            },
            KernelFunction::Permute => match dtype {
                Dtype::F32 => "permute_kernel_f32",
                Dtype::F16 => "permute_kernel_f16",
                Dtype::BF16 => "permute_kernel_bf16",
                _ => "permute_kernel_f32",
            },
            KernelFunction::RepeatKvHeads => match dtype {
                Dtype::F32 => "repeat_kv_heads_kernel_f32",
                Dtype::F16 => "repeat_kv_heads_kernel_f16",
                Dtype::BF16 => "repeat_kv_heads_kernel_bf16",
                _ => "repeat_kv_heads_kernel_f32",
            },
            KernelFunction::Rope => match dtype {
                Dtype::F32 => "rope_kernel_f32",
                Dtype::F16 => "rope_kernel_f16",
                Dtype::BF16 => "rope_kernel_bf16",
                _ => "rope_kernel_f32",
            },
            KernelFunction::RMSNorm => match dtype {
                Dtype::F32 => "rmsnorm_kernel_f32",
                Dtype::F16 => "rmsnorm_kernel_f16",
                Dtype::BF16 => "rmsnorm_kernel_bf16",
                _ => "rmsnorm_kernel_f32",
            },
            KernelFunction::Silu => match dtype {
                Dtype::F32 => "silu_kernel_f32",
                Dtype::F16 => "silu_kernel_f16",
                Dtype::BF16 => "silu_kernel_bf16",
                _ => "silu_kernel_f32",
            },
            KernelFunction::FusedSoftmax => match dtype {
                Dtype::F32 => "sdpa_fused_softmax_f32",
                Dtype::F16 => "sdpa_fused_softmax_f16",
                Dtype::BF16 => "sdpa_fused_softmax_bf16",
                _ => "sdpa_fused_softmax_f32",
            },
            KernelFunction::Arange => match dtype {
                Dtype::F32 => "arange_kernel_f32",
                Dtype::F16 => "arange_kernel_f16",
                Dtype::BF16 => "arange_kernel_bf16",
                _ => "arange_kernel_f32",
            },
            KernelFunction::Ones => match dtype {
                Dtype::F32 => "ones_kernel_f32",
                Dtype::F16 => "ones_kernel_f16",
                Dtype::BF16 => "ones_kernel_bf16",
                _ => "ones_kernel_f32",
            },
            KernelFunction::RandomUniform => match dtype {
                Dtype::F32 => "random_uniform_f32",
                Dtype::F16 => "random_uniform_f16",
                Dtype::BF16 => "random_uniform_bf16",
                _ => "random_uniform_f32",
            },
        }
    }
}
