use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputePipelineState};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::kernels::matmul::mps_matrix_from_buffer;
use crate::metallic::{
    Context, MetalError, Operation, Tensor,
    cache_keys::{MpsMatrixDescriptorKey, MpsSoftMaxKey},
    resource_cache::ResourceCache,
};

use std::mem::size_of;

mod scaled_dot_product_attention_test;

#[derive(Clone, Copy)]
struct SdpaConfig {
    transpose_k: bool,
    reuse_workspace: bool,
    use_mps_softmax: bool,
}

impl SdpaConfig {
    const BASELINE: Self = Self {
        transpose_k: false,
        reuse_workspace: false,
        use_mps_softmax: false,
    };

    const NO_PERMUTE: Self = Self {
        transpose_k: true,
        reuse_workspace: false,
        use_mps_softmax: false,
    };

    const WORKSPACE: Self = Self {
        transpose_k: false,
        reuse_workspace: true,
        use_mps_softmax: false,
    };

    const MPS_SOFTMAX: Self = Self {
        transpose_k: false,
        reuse_workspace: false,
        use_mps_softmax: true,
    };

    const ALL: Self = Self {
        transpose_k: true,
        reuse_workspace: true,
        use_mps_softmax: true,
    };
}

// Public, user-facing, zero-sized struct for the SDPA operation.
pub struct ScaledDotProductAttentionOp;

/// Variant that relies on implicit K transposition through GEMM.
pub struct ScaledDotProductAttentionNoPermuteOp;

/// Variant that reuses a persistent attention workspace between batches.
pub struct ScaledDotProductAttentionWorkspaceOp;

/// Variant that applies attention normalization through `MPSMatrixSoftMax` when supported.
pub struct ScaledDotProductAttentionMpsSoftmaxOp;

/// Variant that combines all optimizations.
pub struct ScaledDotProductAttentionOptimizedOp;

// Internal struct that holds the operation - we'll use existing kernels to implement it
#[allow(dead_code)]
struct ScaledDotProductAttention {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub output: Tensor,
    pub causal: bool,
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
    pub scale: f32,
    pub query_offset: u32,
    pub config: SdpaConfig,
}

fn create_sdpa_operation(
    ctx: &mut Context,
    args: (&Tensor, &Tensor, &Tensor, bool, u32),
    mut cache: Option<&mut ResourceCache>,
    config: SdpaConfig,
) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
    let (q, k, v, causal, query_offset) = args;

    ctx.prepare_tensors_for_active_cmd(&[q, k, v]);

    // Validate dimensions
    if q.dims().len() != 3 || k.dims().len() != 3 || v.dims().len() != 3 {
        return Err(MetalError::InvalidShape("SDPA requires 3D tensors".to_string()));
    }

    let b = q.dims()[0];
    let s_q = q.dims()[1];
    let s_k = k.dims()[1];
    let d = q.dims()[2];

    // Check batch dimension compatibility
    if b != k.dims()[0] || b != v.dims()[0] {
        return Err(MetalError::DimensionMismatch {
            expected: b,
            actual: k.dims()[0].max(v.dims()[0]),
        });
    }

    // Check feature dimension compatibility
    if d != k.dims()[2] {
        return Err(MetalError::DimensionMismatch {
            expected: d,
            actual: k.dims()[2],
        });
    }

    // Check value tensor compatibility
    if s_k != v.dims()[1] || d != v.dims()[2] {
        return Err(MetalError::DimensionMismatch {
            expected: s_k * d,
            actual: v.dims()[1] * v.dims()[2],
        });
    }

    // Calculate scale factor
    let scale = 1.0 / (d as f32).sqrt();

    // Create output tensor
    let out = Tensor::create_tensor_pooled(vec![b, s_q, d], ctx)?;

    let attention = if config.reuse_workspace {
        let buffer = Tensor::create_tensor_pooled(vec![b, s_q, s_k], ctx)?;
        ctx.prepare_tensors_for_active_cmd(&[&buffer]);
        buffer
    } else {
        Tensor::create_tensor_pooled(vec![b, s_q, s_k], ctx)?
    };

    ctx.prepare_tensors_for_active_cmd(&[&attention]);

    let (k_operand, transpose_b) = if config.transpose_k {
        (k.clone(), true)
    } else {
        (k.permute(&[0, 2, 1], ctx)?, false)
    };

    let qk_scaled_result = match cache.as_deref_mut() {
        Some(cache_ref) => ctx.matmul_alpha_beta_with_cache(q, &k_operand, &attention, false, transpose_b, scale, 0.0, cache_ref)?,
        None => ctx.matmul_alpha_beta(q, &k_operand, &attention, false, transpose_b, scale, 0.0)?,
    };

    let softmax_result = {
        let cache_opt = cache.as_deref_mut();
        crate::metallic::kernels::softmax::apply_softmax(
            ctx,
            cache_opt,
            &qk_scaled_result,
            b,
            s_q,
            s_k,
            causal,
            query_offset,
            config.use_mps_softmax,
        )?
    };

    match cache {
        Some(cache_ref) => {
            ctx.matmul_alpha_beta_with_cache(&softmax_result, v, &out, false, false, 1.0, 0.0, cache_ref)?;
        }
        None => {
            ctx.matmul_alpha_beta(&softmax_result, v, &out, false, false, 1.0, 0.0)?;
        }
    };

    // Create a dummy operation since all work is done in this function
    Ok((
        Box::new(ScaledDotProductAttention {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            output: out.clone(),
            causal,
            batch: b,
            seq_q: s_q,
            seq_k: s_k,
            dim: d,
            scale,
            query_offset,
            config,
        }),
        out,
    ))
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for ScaledDotProductAttentionOp {
    // Input arguments for the call - three input tensors + causal flag
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::BASELINE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionNoPermuteOp {
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::NO_PERMUTE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionWorkspaceOp {
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::WORKSPACE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionMpsSoftmaxOp {
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::MPS_SOFTMAX)
    }
}

impl KernelInvocable for ScaledDotProductAttentionOptimizedOp {
    type Args<'a> = (&'a Tensor, &'a Tensor, &'a Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::ALL)
    }
}

// Implement `Operation` for the internal struct.
impl Operation for ScaledDotProductAttention {
    fn encode(
        &self,
        _command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // Since all computation was done in the `new` method of KernelInvocable,
        // this method just returns Ok(())
        Ok(())
    }
}
