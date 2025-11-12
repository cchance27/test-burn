use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use rustc_hash::FxHashMap;

use crate::{Context, Dtype, MetalError, Operation, Tensor, caching::ResourceCache, kernel_lib};

// Rexport KernelManager and Invocable
mod kernel_manager;
pub use kernel_manager::{CustomKernelInvocable, DefaultKernelInvocable, KernelManager, MultiTensorOutput};

pub mod graph_kernel;
pub use graph_kernel::{
    GraphKernel, GraphKernelAccumulator, GraphKernelAxis, GraphKernelDtypePolicy, GraphKernelSignature, GraphKernelTensorDescriptor
};

pub mod backend_registry;
pub use backend_registry::{
    BackendSelection, BackendSelectionReason, KernelBackendKind, KernelBackendOverride, KernelBackendOverrides, KernelBackendRegistry
};

// Export our kernels
pub mod blit_copy;
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
pub mod matmul_gemv_fused2;
pub mod matmul_gemv_qkv_fused;
pub mod matmul_gemv_smallm;
pub mod matmul_gemv_smalln;
pub mod matmul_mlx;
pub mod matmul_mps;
pub mod matmul_q8_canonical;
pub mod matmul_q8_nt;
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
pub mod embedding_lookup;
pub mod sample_topk_topp;
pub mod tensors;

/// Used by our Macro to load kernel source or binary
/// Represents the source of a Metal kernel, either as text source code or as a precompiled
#[derive(Copy, Clone, Debug)]
pub enum KernelSource {
    Text(&'static str),
    Binary(&'static [u8]),
}

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
    Swiglu,
    Tensors,
    MatmulGemv,
    MatmulGemvSmalln,
    MatmulGemmTiled,
    SoftmaxBlock,
    SoftmaxVec,
    SampleTopKTopP,
    EmbeddingLookup,
}

impl KernelLibrary {
    pub fn kernel(&self) -> KernelSource {
        match self {
            KernelLibrary::Cast => kernel_lib!("cast"),
            KernelLibrary::ElemwiseAdd => kernel_lib!("elemwise_add"),
            KernelLibrary::ElemwiseAbs => kernel_lib!("elemwise_abs"),
            KernelLibrary::ElemwiseDiv => kernel_lib!("elemwise_div"),
            KernelLibrary::ElemwiseMul => kernel_lib!("elemwise_mul"),
            KernelLibrary::ElemwiseSub => kernel_lib!("elemwise_sub"),
            KernelLibrary::Gelu => kernel_lib!("gelu"),
            KernelLibrary::KvCacheWrite => kernel_lib!("kv_cache_write"),
            KernelLibrary::KvRearrange => kernel_lib!("kv_rearrange"),
            KernelLibrary::LayerNorm => kernel_lib!("layernorm"),
            KernelLibrary::Permute => kernel_lib!("permute"),
            KernelLibrary::RepeatKvHeads => kernel_lib!("repeat_kv_heads"),
            KernelLibrary::Rope => kernel_lib!("rope"),
            KernelLibrary::RMSNorm => kernel_lib!("rmsnorm"),
            KernelLibrary::Silu => kernel_lib!("silu"),
            KernelLibrary::SoftmaxKernel => kernel_lib!("softmax_kernel"),
            KernelLibrary::Swiglu => kernel_lib!("swiglu"),
            KernelLibrary::Tensors => kernel_lib!("tensors"),
            KernelLibrary::MatmulGemv => kernel_lib!("matmul_gemv"),
            KernelLibrary::MatmulGemvSmalln => kernel_lib!("matmul_gemv_smalln"),
            KernelLibrary::MatmulGemmTiled => kernel_lib!("matmul_gemm_tiled"),
            KernelLibrary::SoftmaxBlock => kernel_lib!("softmax_block"),
            KernelLibrary::SoftmaxVec => kernel_lib!("softmax_vec"),
            KernelLibrary::SampleTopKTopP => kernel_lib!("sample_topk_topp"),
            KernelLibrary::EmbeddingLookup => kernel_lib!("embedding_lookup"),
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
    MatmulGemvSmallM,
    MatmulGemvQkvFused,
    MatmulGemvQ2Fused,
    MatmulQ8Nt,
    MatmulQ8CanonicalLargeN,
    MatmulQ8CanonicalRows16LargeN,
    MatmulGemmTiled,
    SoftmaxBlock,
    SoftmaxVec,
    SampleTopKPartials,
    SampleTopKMergeAndSample,
    SampleTopKFused,
    EmbeddingLookup,
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
            KernelFunction::MatmulGemvQkvFused => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulGemvQ2Fused => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulQ8Nt => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulQ8CanonicalLargeN => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulQ8CanonicalRows16LargeN => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulGemvSmallM => KernelLibrary::MatmulGemv,
            KernelFunction::MatmulGemmTiled => KernelLibrary::MatmulGemmTiled,
            KernelFunction::SoftmaxBlock => KernelLibrary::SoftmaxBlock,
            KernelFunction::SoftmaxVec => KernelLibrary::SoftmaxVec,
            KernelFunction::SampleTopKPartials => KernelLibrary::SampleTopKTopP,
            KernelFunction::SampleTopKMergeAndSample => KernelLibrary::SampleTopKTopP,
            KernelFunction::SampleTopKFused => KernelLibrary::SampleTopKTopP,
            KernelFunction::EmbeddingLookup => KernelLibrary::EmbeddingLookup,
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
            (KernelFunction::MatmulGemv, F32) => "gemv_f32",
            (KernelFunction::MatmulGemv, F16) => "gemv_f16",
            (KernelFunction::MatmulGemvQkvFused, F16) => "gemv_q8_fused3_f16",
            (KernelFunction::MatmulGemvQ2Fused, F16) => "gemv_q8_fused2_f16",
            (KernelFunction::MatmulQ8Nt, F16) => "gemm_q8_nt_f16",
            (KernelFunction::MatmulQ8CanonicalLargeN, F16) => "gemm_q8_canonical_large_n_f16",
            (KernelFunction::MatmulQ8CanonicalRows16LargeN, F16) => "gemm_q8_canonical_large_n_rows16_f16",
            (KernelFunction::MatmulGemvSmallM, F16) => "gemv_q8_rows_f16",
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
            (KernelFunction::SampleTopKPartials, F32) => "sample_topk_partials_f32",
            (KernelFunction::SampleTopKPartials, F16) => "sample_topk_partials_f16",
            (KernelFunction::SampleTopKMergeAndSample, F32) => "sample_topk_merge_and_sample_f32",
            (KernelFunction::SampleTopKMergeAndSample, F16) => "sample_topk_merge_and_sample_f16",
            (KernelFunction::SampleTopKFused, F32) => "sample_topk_fused_f32",
            (KernelFunction::SampleTopKFused, F16) => "sample_topk_fused_f16",
            (KernelFunction::EmbeddingLookup, F32) => "embedding_lookup_kernel_f32",
            (KernelFunction::EmbeddingLookup, F16) => "embedding_lookup_kernel_f16",
            _ => unimplemented!("Kernel function {:?} not implemented for dtype {:?}", self, dtype),
        };

        Ok(name)
    }
}
