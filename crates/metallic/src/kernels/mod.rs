use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use rustc_hash::FxHashMap;

use crate::{
    Context, Dtype, MetalError, Operation, Tensor, encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}, resource_cache::ResourceCache
};

// Rexport KernelManager and Invocable
mod kernel_manager;
pub use kernel_manager::{KernelInvocable, KernelManager};

pub mod graph_kernel;
pub use graph_kernel::{
    GraphKernel, GraphKernelAccumulator, GraphKernelAxis, GraphKernelDtypePolicy, GraphKernelSignature, GraphKernelTensorDescriptor
};

pub mod backend_registry;
pub use backend_registry::{
    BackendSelection, BackendSelectionReason, KernelBackendKind, KernelBackendOverride, KernelBackendOverrides, KernelBackendRegistry
};

// Export our kernels
pub mod elemwise_abs;
pub mod elemwise_add;
pub mod elemwise_div;
pub mod elemwise_mul;
pub mod elemwise_sub;
pub mod gelu;
pub mod kv_cache_write;
pub mod kv_rearrange;
pub mod layernorm;
pub mod matmul_dispatcher;
pub mod matmul_gemm_tiled;
pub mod matmul_gemv;
pub mod matmul_gemv_smalln;
pub mod matmul_mlx;
pub mod matmul_mps;
pub mod sdpa_mps_graph;
pub mod softmax_block;
pub mod softmax_kernel;
pub mod softmax_mps;
pub mod softmax_vec;
pub use matmul_dispatcher::dispatch_op::MatmulDispatchOp;
pub mod permute;
pub mod repeat_kv_heads;
pub mod repeat_kv_heads_graph;
pub mod rmsnorm;
pub mod rope;
pub mod scaled_dot_product_attention;
pub mod silu;

pub mod softmax_dispatcher;
pub mod swiglu;
pub use softmax_dispatcher::SoftmaxDispatchOp;
pub mod tensors;

/// Uniquely identifies a compiled Metal library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelLibrary {
    Cast,
    ElemwiseAdd,
    ElemwiseAbs,
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
    SoftmaxKernel,
    SoftmaxMps,
    Swiglu,
    Tensors,
    MatmulGemv,
    MatmulGemvSmalln,
    MatmulGemmTiled,
    SoftmaxBlock,
    SoftmaxVec,
}

impl KernelLibrary {
    fn source(&self) -> &'static str {
        match self {
            KernelLibrary::Cast => include_str!("cast/kernel.metal"),
            KernelLibrary::ElemwiseAdd => include_str!("elemwise_add/kernel.metal"),
            KernelLibrary::ElemwiseAbs => include_str!("elemwise_abs/kernel.metal"),
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
            KernelLibrary::SoftmaxKernel => include_str!("softmax_kernel/kernel.metal"),
            KernelLibrary::SoftmaxMps => include_str!("softmax_mps/kernel.metal"),
            KernelLibrary::Swiglu => include_str!("swiglu/kernel.metal"),
            KernelLibrary::Tensors => include_str!("tensors/kernel.metal"),
            KernelLibrary::MatmulGemv => include_str!("matmul_gemv/kernel.metal"),
            KernelLibrary::MatmulGemvSmalln => include_str!("matmul_gemv_smalln/kernel.metal"),
            KernelLibrary::MatmulGemmTiled => include_str!("matmul_gemm_tiled/kernel.metal"),
            KernelLibrary::SoftmaxBlock => include_str!("softmax_block/kernel.metal"),
            KernelLibrary::SoftmaxVec => include_str!("softmax_vec/kernel.metal"),
        }
    }
}

/// Uniquely identifies a function within a Metal library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelFunction {
    // Utility
    Noop,
    CastToF16,
    CastFromF16,
    CastToF32,
    CastFromF32,
    ElemwiseAdd,
    ElemwiseAbs,
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
    MatmulGemv,
    MatmulGemvSmallN1,
    MatmulGemvSmallN2,
    MatmulGemvSmallN4,
    MatmulGemvSmallN8,
    MatmulGemvSmallN16,
    MatmulGemmTiled,
    SoftmaxBlock,
    SoftmaxVec,
}

impl KernelFunction {
    fn library(&self) -> KernelLibrary {
        match self {
            KernelFunction::Noop => KernelLibrary::Tensors,
            KernelFunction::CastToF16 | KernelFunction::CastFromF16 | KernelFunction::CastToF32 | KernelFunction::CastFromF32 => {
                KernelLibrary::Cast
            }
            KernelFunction::ElemwiseAdd | KernelFunction::ElemwiseBroadcastAdd => KernelLibrary::ElemwiseAdd,
            KernelFunction::ElemwiseAbs => KernelLibrary::ElemwiseAbs,
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
            KernelFunction::FusedSoftmax => KernelLibrary::SoftmaxKernel,
            KernelFunction::SwigluFusedActivation => KernelLibrary::Swiglu,
            KernelFunction::Arange | KernelFunction::Ones | KernelFunction::RandomUniform => KernelLibrary::Tensors,
            KernelFunction::MatmulGemv => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulGemvSmallN1 => KernelLibrary::MatmulGemvSmalln,
            KernelFunction::MatmulGemvSmallN2 => KernelLibrary::MatmulGemvSmalln,
            KernelFunction::MatmulGemvSmallN4 => KernelLibrary::MatmulGemvSmalln,
            KernelFunction::MatmulGemvSmallN8 => KernelLibrary::MatmulGemvSmalln,
            KernelFunction::MatmulGemvSmallN16 => KernelLibrary::MatmulGemvSmalln,
            KernelFunction::MatmulGemmTiled => KernelLibrary::MatmulGemmTiled,
            KernelFunction::SoftmaxBlock => KernelLibrary::SoftmaxBlock,
            KernelFunction::SoftmaxVec => KernelLibrary::SoftmaxVec,
        }
    }

    fn name_for_dtype(&self, dtype: Dtype) -> Result<&'static str, MetalError> {
        use Dtype::*;

        let name = match (self, dtype) {
            (KernelFunction::Noop, F32) => "noop_kernel_f32",
            (KernelFunction::Noop, F16) => "noop_kernel_f16",
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
            (KernelFunction::ElemwiseAbs, F32) => "abs_kernel_f32",
            (KernelFunction::ElemwiseAbs, F16) => "abs_kernel_f16",
            (KernelFunction::ElemwiseBroadcastAdd, F32) => "broadcast_add_kernel_f32",
            (KernelFunction::ElemwiseBroadcastAdd, F16) => "broadcast_add_kernel_f16",
            (KernelFunction::ElemwiseDiv, F32) => "div_kernel_f32",
            (KernelFunction::ElemwiseDiv, F16) => "div_kernel_f32",
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
            (KernelFunction::MatmulGemv, F32) => "gemv_f32",
            (KernelFunction::MatmulGemv, F16) => "gemv_f16",
            (KernelFunction::MatmulGemvSmallN1, F16) => "gemv_n1_f16",
            (KernelFunction::MatmulGemvSmallN1, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemvSmallN1",
                });
            }
            (KernelFunction::MatmulGemvSmallN2, F16) => "gemv_n2_f16",
            (KernelFunction::MatmulGemvSmallN2, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemvSmallN2",
                });
            }
            (KernelFunction::MatmulGemvSmallN4, F16) => "gemv_n4_f16",
            (KernelFunction::MatmulGemvSmallN4, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemvSmallN4",
                });
            }
            (KernelFunction::MatmulGemvSmallN8, F16) => "gemv_n8_f16",
            (KernelFunction::MatmulGemvSmallN8, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemvSmallN8",
                });
            }
            (KernelFunction::MatmulGemvSmallN16, F16) => "gemv_n16_f16",
            (KernelFunction::MatmulGemvSmallN16, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemvSmallN16",
                });
            }
            (KernelFunction::MatmulGemmTiled, F16) => "gemm_tiled_f16",
            (KernelFunction::MatmulGemmTiled, F32) => {
                return Err(MetalError::UnsupportedDtype {
                    dtype: Dtype::F32,
                    operation: "MatmulGemmTiled",
                });
            }
            (KernelFunction::SoftmaxBlock, F16) => "block_softmax_f16",
            (KernelFunction::SoftmaxBlock, F32) => "block_softmax_f32",
            (KernelFunction::SoftmaxVec, F16) => "vec_softmax_f16",
            (KernelFunction::SoftmaxVec, F32) => "vec_softmax_f32",
        };

        Ok(name)
    }
}
