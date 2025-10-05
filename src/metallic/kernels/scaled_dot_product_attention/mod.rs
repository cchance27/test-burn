use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputePipelineState};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{
    Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, cache_keys::SdpaKey, resource_cache::ResourceCache,
};

#[cfg(test)]
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
    args: (&Tensor<T>, &Tensor<T>, &Tensor<T>, bool, u32, u32),
    mut cache: Option<&mut ResourceCache>,
    config: SdpaConfig,
) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
    let (q, k, v, causal, query_offset, group_size_raw) = args;

    if group_size_raw == 0 {
        return Err(MetalError::InvalidShape(
            "scaled_dot_product_attention requires a non-zero group size".to_string(),
        ));
    }
    let group_size = group_size_raw as usize;

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
    };
    let scale = if let Some(cache_ref) = cache.as_mut() {
        cache_ref
            .get_or_create_sdpa(sdpa_descriptor.batch, sdpa_descriptor.dim, sdpa_descriptor.dtype)
            .scale
    } else {
        compute_sdpa_scale(d)
    };

    if group_size > 1 && s_q % group_size != 0 {
        return Err(MetalError::InvalidShape(format!(
            "SDPA group size {} requires query rows {} to be divisible",
            group_size, s_q
        )));
    }

    let tokens_total = if group_size > 1 { s_q / group_size } else { s_q };
    if tokens_total == 0 {
        return Err(MetalError::InvalidShape("SDPA requires at least one query row".to_string()));
    }

    let query_offset_rows =
        usize::try_from(query_offset).map_err(|_| MetalError::InvalidShape("SDPA query offset exceeds usize range".to_string()))?;
    if group_size > 1 && query_offset_rows % group_size != 0 {
        return Err(MetalError::InvalidShape(format!(
            "SDPA query offset {} must be divisible by group size {}",
            query_offset, group_size
        )));
    }

    let mut seq_len_delta_tokens = s_k;

    if causal {
        let workspace_key = ctx.sdpa_workspace_key_for(k);
        if query_offset == 0 {
            ctx.reset_sdpa_workspace(workspace_key);
        }
        seq_len_delta_tokens = ctx.sdpa_seq_delta(workspace_key, sdpa_descriptor.clone(), tokens_total, s_k);
    }

    if seq_len_delta_tokens == 0 {
        seq_len_delta_tokens = tokens_total;
    }

    if seq_len_delta_tokens > tokens_total {
        seq_len_delta_tokens = tokens_total;
    }

    let rows_to_process_tokens = seq_len_delta_tokens;

    let rows_to_process = rows_to_process_tokens
        .checked_mul(group_size)
        .ok_or_else(|| MetalError::InvalidShape("SDPA rows_to_process overflow".to_string()))?;
    if rows_to_process == 0 {
        return Err(MetalError::InvalidShape("SDPA requires at least one query row".to_string()));
    }
    if rows_to_process > s_q {
        return Err(MetalError::InvalidShape(format!(
            "SDPA computed rows_to_process {} exceeds available queries {}",
            rows_to_process, s_q
        )));
    }

    let row_offset_tokens = tokens_total.saturating_sub(rows_to_process_tokens);
    let row_offset = row_offset_tokens
        .checked_mul(group_size)
        .ok_or_else(|| MetalError::InvalidShape("SDPA row offset overflow".to_string()))?;

    let q_active = if row_offset == 0 && rows_to_process == s_q {
        q.clone()
    } else {
        let seq_stride = *q.strides.get(1).unwrap_or(&d);
        let elem_size = q.dtype.size_bytes();
        let offset_bytes = row_offset
            .checked_mul(seq_stride)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| MetalError::InvalidShape("SDPA query slice offset exceeds representable range".to_string()))?;

        let mut view = q.clone();
        view.offset = view
            .offset
            .checked_add(offset_bytes)
            .ok_or_else(|| MetalError::InvalidShape("SDPA query view offset exceeds representable range".to_string()))?;
        view.dims = vec![b, rows_to_process, d];
        view
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

    let (k_operand, transpose_b) = if config.transpose_k {
        (k.clone(), true)
    } else {
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
    // Offset the caller-provided base by the number of rows we skipped so causal masks
    // continue to align with the first active query row within the sliced window.
    let adjusted_query_offset = query_offset.checked_add(row_offset_u32).ok_or_else(|| {
        MetalError::InvalidShape(format!(
            "SDPA query offset {query_offset} with row offset {row_offset} exceeds u32::MAX"
        ))
    })?;

    let softmax_result = {
        let cache_opt = cache.as_deref_mut();
        crate::metallic::kernels::softmax::apply_softmax(
            ctx,
            cache_opt,
            &qk_scaled_result,
            b,
            rows_to_process,
            s_k,
            causal,
            adjusted_query_offset,
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
impl KernelInvocable for ScaledDotProductAttentionOp {
    // Input arguments for the call - three input tensors + causal flag
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32, u32);

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

impl KernelInvocable for ScaledDotProductAttentionNoPermuteOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32, u32);

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

impl KernelInvocable for ScaledDotProductAttentionWorkspaceOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32, u32);

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

impl KernelInvocable for ScaledDotProductAttentionMpsSoftmaxOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32, u32);

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

impl KernelInvocable for ScaledDotProductAttentionOptimizedOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32, u32);

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

// Implement `Operation` for the internal struct.
impl<T: TensorElement> Operation for ScaledDotProductAttention<T> {
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
