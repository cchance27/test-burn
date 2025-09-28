use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputePipelineState};
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixSoftMax};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{
    cache_keys::{MpsMatrixDescriptorKey, MpsSoftMaxKey},
    resource_cache::ResourceCache,
    Context, MetalError, Operation, Tensor,
};
use crate::metallic::kernels::matmul::mps_matrix_from_buffer;

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
    args: (Tensor, Tensor, Tensor, bool, u32),
    mut cache: Option<&mut ResourceCache>,
    config: SdpaConfig,
) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
    let (mut q, mut k, mut v, causal, query_offset) = args;

    ctx.prepare_tensors_for_active_cmd(&mut [&mut q, &mut k, &mut v]);

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

    let workspace = if config.reuse_workspace {
        Some(Tensor::create_tensor_pooled(vec![s_q, s_k], ctx)?)
    } else {
        None
    };

    // Process each batch separately to work with 2D matmul operations
    for i in 0..b {
        // Get batch slices for each tensor
        let q_i = q.get_batch(i)?; // [s_q, d]
        let k_i = k.get_batch(i)?; // [s_k, d]
        let v_i = v.get_batch(i)?; // [s_k, d]
        let out_i = out.get_batch(i)?; // [s_q, d]

        let (k_operand, transpose_b) = if config.transpose_k {
            (k_i.clone(), true)
        } else {
            (k_i.permute(&[1, 0], ctx)?, false)
        };

        let attention_seed = match workspace.as_ref() {
            Some(buffer) => buffer.clone(),
            None => Tensor::zeros(vec![s_q, s_k], ctx, true)?,
        };

        // Perform matmul with scaling in one operation: [s_q, d] @ [d, s_k] = [s_q, s_k] with alpha=scale
        let qk_scaled_result = ctx.matmul_alpha_beta(&q_i, &k_operand, &attention_seed, false, transpose_b, scale, 0.0)?; // [s_q, s_k]

        let use_mps_softmax = config.use_mps_softmax && !causal && query_offset == 0;

        let softmax_result = if use_mps_softmax {
            if let Some(cache_ref) = cache.as_mut() {
                apply_mps_softmax(ctx, *cache_ref, &qk_scaled_result, s_q, s_k)?;
                qk_scaled_result.clone()
            } else {
                ctx.call::<crate::metallic::kernels::softmax::SoftmaxOp>((
                    qk_scaled_result,
                    s_q as u32,
                    s_k as u32,
                    causal as u32,
                    query_offset,
                ))?
            }
        } else {
            ctx.call::<crate::metallic::kernels::softmax::SoftmaxOp>((
                qk_scaled_result,
                s_q as u32,
                s_k as u32,
                causal as u32,
                query_offset,
            ))?
        };

        // softmax_result x V -> out (for this batch)
        // [s_q, s_k] @ [s_k, d] = [s_q, d]
        ctx.matmul_alpha_beta(&softmax_result, &v_i, &out_i, false, false, 1.0, 0.0)?;
        // [s_q, d]
    }

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

fn apply_mps_softmax(ctx: &mut Context, cache: &mut ResourceCache, attn: &Tensor, rows: usize, columns: usize) -> Result<(), MetalError> {
    let mut attn_for_prepare = attn.clone();
    ctx.prepare_tensors_for_active_cmd(&mut [&mut attn_for_prepare]);

    let descriptor_key = MpsMatrixDescriptorKey {
        rows,
        columns,
        row_bytes: columns * size_of::<f32>(),
    };
    let descriptor = cache.get_or_create_descriptor(descriptor_key, &ctx.device)?;
    let softmax_key = MpsSoftMaxKey { rows, columns };
    let softmax = cache.get_or_create_softmax(softmax_key, &ctx.device)?;

    let command_buffer = ctx.active_command_buffer_mut()?;
    let op = MpsSoftmaxOperation {
        attn: attn.clone(),
        descriptor,
        softmax,
    };
    command_buffer.record(&op, cache)?;
    ctx.mark_tensor_pending(attn);
    Ok(())
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for ScaledDotProductAttentionOp {
    // Input arguments for the call - three input tensors + causal flag
    type Args = (Tensor, Tensor, Tensor, bool, u32); // (q, k, v, causal, query_offset)

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::BASELINE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionNoPermuteOp {
    type Args = (Tensor, Tensor, Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::NO_PERMUTE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionWorkspaceOp {
    type Args = (Tensor, Tensor, Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::WORKSPACE)
    }
}

impl KernelInvocable for ScaledDotProductAttentionMpsSoftmaxOp {
    type Args = (Tensor, Tensor, Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::MPS_SOFTMAX)
    }
}

impl KernelInvocable for ScaledDotProductAttentionOptimizedOp {
    type Args = (Tensor, Tensor, Tensor, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
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

struct MpsSoftmaxOperation {
    attn: Tensor,
    descriptor: Retained<MPSMatrixDescriptor>,
    softmax: Retained<MPSMatrixSoftMax>,
}

impl Operation for MpsSoftmaxOperation {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let attn_matrix = mps_matrix_from_buffer(&self.attn.buf, self.attn.offset, &self.descriptor);
        unsafe {
            self.softmax
                .encodeToCommandBuffer_inputMatrix_resultMatrix(command_buffer, &attn_matrix, &attn_matrix);
        }
        Ok(())
    }
}
