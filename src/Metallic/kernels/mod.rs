use crate::metallic::{
    encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state},
    resource_cache::ResourceCache,
    Context, MetalError, Operation, Tensor,
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
            KernelFunction::RepeatKvHeads | KernelFunction::RepeatKvHeadsStep => KernelLibrary::RepeatKvHeads,
            KernelFunction::Rope => KernelLibrary::Rope,
            KernelFunction::RMSNorm => KernelLibrary::RMSNorm,
            KernelFunction::Silu => KernelLibrary::Silu,
            KernelFunction::FusedSoftmax => KernelLibrary::Softmax,
            KernelFunction::Arange | KernelFunction::Ones | KernelFunction::RandomUniform => KernelLibrary::Tensors,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            KernelFunction::ElemwiseAdd => "add_kernel",
            KernelFunction::ElemwiseBroadcastAdd => "broadcast_add_kernel",
            KernelFunction::ElemwiseDiv => "div_kernel",
            KernelFunction::ElemwiseMul => "mul_kernel",
            KernelFunction::ElemwiseSub => "sub_kernel",
            KernelFunction::FusedQkvBiasSplit => "fused_qkv_bias_split",
            KernelFunction::Gelu => "gelu_kernel",
            KernelFunction::KvRearrange => "kv_rearrange_kernel",
            KernelFunction::LayerNorm => "layernorm_kernel",
            KernelFunction::Permute => "permute_kernel",
            KernelFunction::RepeatKvHeads => "repeat_kv_heads_kernel",
            KernelFunction::RepeatKvHeadsStep => "repeat_kv_heads_step_kernel",
            KernelFunction::Rope => "rope_kernel",
            KernelFunction::RMSNorm => "rmsnorm_kernel",
            KernelFunction::Silu => "silu_kernel",
            KernelFunction::FusedSoftmax => "sdpa_fused_softmax",
            KernelFunction::Arange => "arange_kernel",
            KernelFunction::Ones => "ones_kernel",
            KernelFunction::RandomUniform => "random_uniform",
        }
    }
}
