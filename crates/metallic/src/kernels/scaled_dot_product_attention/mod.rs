use metallic_instrumentation::{MetricEvent, record_metric_async};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

use super::{DefaultKernelInvocable, KernelBackendKind, KernelFunction};
use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, caching::ResourceCache, kernels::{scaled_dot_product_attention::cache::SdpaKey, sdpa_mps_graph::SdpaMpsGraphOp, softmax_mps::cache::SeqKBucket}
};

#[cfg(test)]
mod scaled_dot_product_attention_test;

pub mod cache;

#[derive(Clone, Copy)]
struct SdpaConfig {
    transpose_k: bool,
    reuse_workspace: bool,
}

impl SdpaConfig {
    const BASELINE: Self = Self {
        transpose_k: true, // Prefer logical transpose by default to avoid materialized permutes
        reuse_workspace: false,
    };

    const NO_PERMUTE: Self = Self {
        transpose_k: true, // Same as baseline now - both prefer logical transpose
        reuse_workspace: false,
    };

    const WORKSPACE: Self = Self {
        transpose_k: false,
        reuse_workspace: true,
    };

    // Note: This configuration is now handled by the new softmax dispatcher
    const MPS_SOFTMAX: Self = Self {
        transpose_k: false,
        reuse_workspace: false,
    };

    const ALL: Self = Self {
        transpose_k: true,
        reuse_workspace: true,
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

/// Dispatches between legacy and graph-backed SDPA implementations.
pub struct ScaledDotProductAttentionDispatchOp;

// Internal struct that holds the operation - we'll use existing kernels to implement it
#[allow(dead_code)]
struct ScaledDotProductAttention<T: TensorElement> {
    pub q: Tensor<T>,
    pub k: Tensor<T>,
    pub v: Tensor<T>,
    pub output: Tensor<T>,
    pub causal: bool,
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
    pub scale: f32,
    pub query_offset: u32,
    pub seq_len_delta: usize,
    pub config: SdpaConfig,
}

fn create_sdpa_operation<T: TensorElement>(
    ctx: &mut Context<T>,
    args: (&Tensor<T>, &Tensor<T>, &Tensor<T>, bool, u32),
    mut cache: Option<&mut ResourceCache>,
    config: SdpaConfig,
) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
    let (q, k, v, causal, query_offset) = args;

    ctx.prepare_tensors_for_active_cmd(&[q, k, v])?;

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

    // Calculate scale factor, reusing the cached SDPA descriptor when available.
    let sdpa_descriptor = SdpaKey {
        batch: b,
        dim: d,
        dtype: q.dtype,
        causal,
        seq_k_bucket: SeqKBucket::from(s_k),
        transpose_k: config.transpose_k,
    };
    let scale = if let Some(cache_ref) = cache.as_mut() {
        cache_ref
            .get_or_create_sdpa_full(
                sdpa_descriptor.batch,
                sdpa_descriptor.dim,
                sdpa_descriptor.dtype,
                sdpa_descriptor.causal,
                s_k,
                sdpa_descriptor.transpose_k,
            )
            .scale
    } else {
        compute_sdpa_scale(d)
    };

    let mut seq_len_delta = s_k;
    if causal {
        let workspace_key = ctx.sdpa_workspace_key_for(k);
        seq_len_delta = ctx.sdpa_seq_delta(workspace_key, sdpa_descriptor.clone(), s_q, s_k);
    }

    let is_incremental_decode = causal && seq_len_delta == 1 && s_q > 1 && s_k > 1;
    let mut rows_to_process = if is_incremental_decode { 1 } else { s_q };

    if rows_to_process == 0 && s_q > 0 {
        rows_to_process = s_q;
    }
    let row_offset = s_q.saturating_sub(rows_to_process);

    let q_active = if row_offset == 0 && rows_to_process == s_q {
        q.clone()
    } else {
        if b > 1 {
            return Err(MetalError::OperationNotSupported(
                "Incremental SDPA with batch size > 1 is not yet supported.".to_string(),
            ));
        }
        let q_2d = q.reshape(vec![s_q, d])?;
        let sliced_q = q_2d.slice(row_offset..s_q)?;
        sliced_q.reshape(vec![b, rows_to_process, d])?
    };

    // Create output tensor
    let out = Tensor::new(vec![b, rows_to_process, d], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

    let attention = if config.reuse_workspace {
        let buffer = Tensor::new(vec![b, rows_to_process, s_k], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        ctx.prepare_tensors_for_active_cmd(&[&buffer])?;
        buffer
    } else {
        Tensor::new(vec![b, rows_to_process, s_k], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?
    };

    ctx.prepare_tensors_for_active_cmd(&[&attention])?;

    // For the QK^T matmul, prefer logical transpose via stride manipulation over materialized permute
    // when the backend supports it (MLX does, MPS does, etc.)
    let (k_operand, transpose_b) = if config.transpose_k {
        // Use logical transpose by swapping dimensions and strides - no data movement
        (k.clone(), true)
    } else {
        // Fallback to materialized transpose - requires copying data
        (k.permute(&[0, 2, 1], ctx)?, false)
    };

    let qk_scaled_result = match cache.as_deref_mut() {
        Some(cache_ref) => {
            ctx.matmul_alpha_beta_with_cache(&q_active, &k_operand, &attention, false, transpose_b, scale, 0.0, cache_ref)?
        }
        None => ctx.matmul_alpha_beta(&q_active, &k_operand, &attention, false, transpose_b, scale, 0.0)?,
    };

    let row_offset_u32 = u32::try_from(row_offset)
        .map_err(|_| MetalError::InvalidShape(format!("SDPA row offset {row_offset} exceeds representable query offset range")))?;
    let adjusted_query_offset = query_offset.checked_add(row_offset_u32).ok_or_else(|| {
        MetalError::InvalidShape(format!(
            "SDPA query offset {query_offset} with row offset {row_offset} exceeds u32::MAX"
        ))
    })?;

    let softmax_result = {
        // Use the softmax dispatcher to select optimal backend/variant
        if let Some(cache_ref) = cache.as_mut() {
            ctx.call_with_cache::<crate::kernels::softmax_dispatcher::dispatch_op::SoftmaxDispatchOp>(
                (&qk_scaled_result, causal, adjusted_query_offset),
                cache_ref,
            )?
        } else {
            ctx.call::<crate::kernels::softmax_dispatcher::dispatch_op::SoftmaxDispatchOp>((
                &qk_scaled_result,
                causal,
                adjusted_query_offset,
            ))?
        }
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
            seq_q: rows_to_process,
            seq_k: s_k,
            dim: d,
            scale,
            query_offset: adjusted_query_offset,
            seq_len_delta: rows_to_process,
            config,
        }),
        out,
    ))
}

fn compute_sdpa_scale(dim: usize) -> f32 {
    let dim_f32 = dim as f32;
    let scale = 1.0 / dim_f32.sqrt();

    if scale.is_infinite() || scale.is_nan() {
        1.0
    } else {
        scale.clamp(1e-6, 1e6)
    }
}

// Implement `KernelInvocable` for the public struct.
impl DefaultKernelInvocable for ScaledDotProductAttentionOp {
    // Input arguments for the call - three input tensors + causal flag
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::BASELINE)
    }
}

impl DefaultKernelInvocable for ScaledDotProductAttentionNoPermuteOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::NO_PERMUTE)
    }
}

impl DefaultKernelInvocable for ScaledDotProductAttentionWorkspaceOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::WORKSPACE)
    }
}

impl DefaultKernelInvocable for ScaledDotProductAttentionMpsSoftmaxOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::MPS_SOFTMAX)
    }
}

impl DefaultKernelInvocable for ScaledDotProductAttentionOptimizedOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        create_sdpa_operation(ctx, args, cache, SdpaConfig::ALL)
    }
}

impl DefaultKernelInvocable for ScaledDotProductAttentionDispatchOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let selection = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy);

        let profiler_backend = match selection.backend {
            KernelBackendKind::Legacy => "Metal",
            KernelBackendKind::Graph => "MPSGraph",
        };
        ctx.override_pending_gpu_backend(profiler_backend);

        let result = match selection.backend {
            KernelBackendKind::Legacy => ScaledDotProductAttentionOptimizedOp::new(ctx, args, None, cache),
            KernelBackendKind::Graph => SdpaMpsGraphOp::new(ctx, args, None, cache),
        }?;

        record_metric_async!(MetricEvent::KernelBackendSelected {
            op_name: "sdpa".to_string(),
            backend: selection.backend.as_str().to_string(),
            reason: selection.reason.as_str().to_string(),
        });

        Ok(result)
    }
}

// Implement `Operation` for the internal struct.
impl<T: TensorElement> Operation for ScaledDotProductAttention<T> {
    fn encode(&self, _command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        // Since all computation was done in the `new` method of DefaultKernelInvocable,
        // this method just returns Ok(())
        Ok(())
    }
}
