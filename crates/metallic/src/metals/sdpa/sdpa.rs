use std::sync::OnceLock;

use crate::{
    MetalError, foundry::{Foundry, storage::Pooled, tensor::Tensor}, metals::{
        gemv::{GemvColMajor, GemvParams, GemvRowMajor}, softmax::Softmax
    }, tensor::{TensorInit, dtypes::F16 as F16Dtype}, types::TensorArg
};

/// SDPA backend selection for future MLX comparison testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdpaBackend {
    /// Auto-select: Gemv for decode, unrolled loop for prefill (default, matches Legacy)
    Auto,
    /// Force unrolled Gemv loop for all cases (correctness testing)
    ForceUnroll,
    /// Force MLX/Matmul backend when available (future perf testing)
    ForceMlx,
}

fn sdpa_backend_override() -> SdpaBackend {
    static BACKEND: OnceLock<SdpaBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| match std::env::var("METALLIC_SDPA_BACKEND").ok().as_deref() {
        Some("unroll") | Some("loop") => SdpaBackend::ForceUnroll,
        Some("mlx") | Some("matmul") => SdpaBackend::ForceMlx,
        _ => SdpaBackend::Auto,
    })
}

/// Scaled Dot Product Attention using Chained Dispatch (Gemv -> Softmax -> Gemv).
///
/// Implements the same optimization as Legacy: for incremental decode (query_offset > 0),
/// only the newest query row is processed, enabling efficient Gemv kernels.
pub fn scaled_dot_product_attention(
    foundry: &mut Foundry,
    q: &Tensor<F16Dtype, impl crate::foundry::storage::StorageState>,
    k: &Tensor<F16Dtype, impl crate::foundry::storage::StorageState>,
    v: &Tensor<F16Dtype, impl crate::foundry::storage::StorageState>,
    causal: bool,
    query_offset: u32,
) -> Result<Tensor<F16Dtype, Pooled>, MetalError> {
    scaled_dot_product_attention_impl(foundry, q, k, v, causal, query_offset)
}

fn scaled_dot_product_attention_impl<S1, S2, S3>(
    foundry: &mut Foundry,
    q: &Tensor<F16Dtype, S1>,
    k: &Tensor<F16Dtype, S2>,
    v: &Tensor<F16Dtype, S3>,
    causal: bool,
    query_offset: u32,
) -> Result<Tensor<F16Dtype, Pooled>, MetalError>
where
    S1: crate::foundry::storage::StorageState,
    S2: crate::foundry::storage::StorageState,
    S3: crate::foundry::storage::StorageState,
{
    // Validation
    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();

    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
        return Err(MetalError::InvalidShape("SDPA requires 3D tensors".to_string()));
    }

    let batch = q_dims[0];
    let seq_q = q_dims[1];
    let dim = q_dims[2];
    let seq_k = k_dims[1];

    if k_dims[0] != batch || k_dims[2] != dim {
        return Err(MetalError::DimensionMismatch {
            expected: dim,
            actual: k_dims[2],
        });
    }

    if v_dims[0] != batch || v_dims[1] != seq_k || v_dims[2] != dim {
        return Err(MetalError::DimensionMismatch {
            expected: seq_k,
            actual: v_dims[1],
        });
    }

    let scale = 1.0 / (dim as f32).sqrt();
    let backend = sdpa_backend_override();

    // ============================================================================
    // Incremental Decode Optimization (matches Legacy behavior)
    // ============================================================================
    // Legacy logic (from scaled_dot_product_attention/mod.rs):
    //   is_incremental_decode = causal && seq_len_delta == 1 && s_q > 1 && s_k > 1
    //   rows_to_process = if is_incremental_decode { 1 } else { s_q }
    //
    // We use `query_offset > 0` as a proxy for `seq_len_delta == 1` since:
    //   - query_offset > 0 means we're appending to an existing KV cache (decode phase)
    //   - query_offset == 0 with seq_q > 1 means initial prefill
    //
    // This optimization processes only the newest query row during decode,
    // reducing M to 1 and enabling efficient Gemv kernels.

    let is_incremental_decode = causal && query_offset > 0 && seq_q > 0 && seq_k > 1;
    let rows_to_process = if is_incremental_decode && backend == SdpaBackend::Auto {
        1
    } else {
        seq_q
    };
    let row_offset = seq_q.saturating_sub(rows_to_process);

    // Determine effective query offset for softmax masking
    let effective_query_offset = query_offset + row_offset as u32;

    // Create output/intermediate tensors for the rows we'll actually process
    let scores = Tensor::<F16Dtype, Pooled>::new(foundry, vec![batch, rows_to_process, seq_k], TensorInit::Uninitialized)?;

    let out = Tensor::<F16Dtype, Pooled>::new(foundry, vec![batch, rows_to_process, dim], TensorInit::Uninitialized)?;

    // ============================================================================
    // Dispatch Logic
    // ============================================================================

    // Use optimized GPU Batching if rows_to_process == 1 (decode or single query)
    let use_gpu_batching = rows_to_process == 1 && backend != SdpaBackend::ForceUnroll;

    // TODO: When MLX is ported, add: backend == SdpaBackend::ForceMlx -> use MLX
    if backend == SdpaBackend::ForceMlx {
        return Err(MetalError::OperationNotSupported(
            "MLX backend for SDPA not yet implemented. Use METALLIC_SDPA_BACKEND=auto or unroll".to_string(),
        ));
    }

    if use_gpu_batching {
        // Fast path: GPU Batching with Gemv (M=1)

        // Compute Q slice offset for incremental decode
        let q_row_byte_offset = row_offset * q.strides()[1] * 2;
        let q_slice = q.view(vec![batch, 1, dim], q.strides().to_vec(), q.offset() + q_row_byte_offset);

        // Step 1: Q * K^T -> Scores
        let params_qk = GemvParams {
            k: dim as u32,
            n: seq_k as u32,
            blocks_per_k: 1,
            weights_per_block: dim as u32,
            batch: batch as u32,
            stride_x: q.strides()[0] as u32,
            stride_y: scores.strides()[0] as u32,
            stride_a: k.strides()[0] as u32,
            stride_w: dim as u32,
            stride_scale: 0,
        };

        let kernel_qk = GemvColMajor::new(
            &TensorArg::from_tensor(k),
            &TensorArg::from_tensor(&q_slice),
            &TensorArg::from_tensor(&scores),
            params_qk,
        )
        .with_alpha(scale);
        foundry.run(&kernel_qk)?;

        // Step 2: Softmax
        let softmax_kernel = Softmax::new(
            &TensorArg::from_tensor(&scores),
            &TensorArg::from_tensor(&scores),
            batch as u32 * rows_to_process as u32,
            rows_to_process as u32,
            seq_k as u32,
            causal,
            effective_query_offset,
        );
        foundry.run(&softmax_kernel)?;

        // Step 3: Probs * V -> Output
        let params_av = GemvParams {
            k: seq_k as u32,
            n: dim as u32,
            blocks_per_k: 1,
            weights_per_block: seq_k as u32,
            batch: batch as u32,
            stride_x: scores.strides()[0] as u32,
            stride_y: out.strides()[0] as u32,
            stride_a: v.strides()[0] as u32,
            stride_w: dim as u32,
            stride_scale: 0,
        };

        let kernel_av = GemvRowMajor::new(
            &TensorArg::from_tensor(v),
            &TensorArg::from_tensor(&scores),
            &TensorArg::from_tensor(&out),
            params_av,
        );
        foundry.run(&kernel_av)?;
    } else {
        // Slow path: Unrolled loop for prefill (rows_to_process > 1)
        // This matches Legacy behavior when is_incremental_decode is false

        for b in 0..batch {
            let q_batch_offset = b * q.strides()[0] * 2;
            let k_batch_offset = b * k.strides()[0] * 2;
            let scores_batch_offset = b * scores.strides()[0] * 2;

            for q_idx in 0..rows_to_process {
                let actual_q_row = row_offset + q_idx;
                let q_row_offset = actual_q_row * q.strides()[1] * 2;
                let scores_row_offset = q_idx * scores.strides()[1] * 2;

                let q_slice = q.view(vec![1, 1, dim], q.strides().to_vec(), q.offset() + q_batch_offset + q_row_offset);
                let scores_slice = scores.view(
                    vec![1, 1, seq_k],
                    scores.strides().to_vec(),
                    scores.offset() + scores_batch_offset + scores_row_offset,
                );
                let k_slice = k.view(vec![1, seq_k, dim], k.strides().to_vec(), k.offset() + k_batch_offset);

                let params_qk = GemvParams {
                    k: dim as u32,
                    n: seq_k as u32,
                    blocks_per_k: 1,
                    weights_per_block: dim as u32,
                    batch: 1,
                    stride_x: 0,
                    stride_y: 0,
                    stride_a: 0,
                    stride_w: dim as u32,
                    stride_scale: 0,
                };

                let kernel_qk = GemvColMajor::new(
                    &TensorArg::from_tensor(&k_slice),
                    &TensorArg::from_tensor(&q_slice),
                    &TensorArg::from_tensor(&scores_slice),
                    params_qk,
                )
                .with_alpha(scale);
                foundry.run(&kernel_qk)?;
            }
        }

        // Step 2: Softmax (can be batched across all rows)
        let softmax_kernel = Softmax::new(
            &TensorArg::from_tensor(&scores),
            &TensorArg::from_tensor(&scores),
            batch as u32 * rows_to_process as u32,
            rows_to_process as u32,
            seq_k as u32,
            causal,
            effective_query_offset,
        );
        foundry.run(&softmax_kernel)?;

        // Step 3: Probs * V -> Output (unrolled)
        for b in 0..batch {
            let scores_batch_offset = b * scores.strides()[0] * 2;
            let v_batch_offset = b * v.strides()[0] * 2;
            let out_batch_offset = b * out.strides()[0] * 2;

            for q_idx in 0..rows_to_process {
                let scores_row_offset = q_idx * scores.strides()[1] * 2;
                let out_row_offset = q_idx * out.strides()[1] * 2;

                let scores_slice = scores.view(
                    vec![1, 1, seq_k],
                    scores.strides().to_vec(),
                    scores.offset() + scores_batch_offset + scores_row_offset,
                );
                let out_slice = out.view(
                    vec![1, 1, dim],
                    out.strides().to_vec(),
                    out.offset() + out_batch_offset + out_row_offset,
                );
                let v_slice = v.view(vec![1, seq_k, dim], v.strides().to_vec(), v.offset() + v_batch_offset);

                let params_av = GemvParams {
                    k: seq_k as u32,
                    n: dim as u32,
                    blocks_per_k: 1,
                    weights_per_block: seq_k as u32,
                    batch: 1,
                    stride_x: 0,
                    stride_y: 0,
                    stride_a: 0,
                    stride_w: dim as u32,
                    stride_scale: 0,
                };

                let kernel_av = GemvRowMajor::new(
                    &TensorArg::from_tensor(&v_slice),
                    &TensorArg::from_tensor(&scores_slice),
                    &TensorArg::from_tensor(&out_slice),
                    params_av,
                );
                foundry.run(&kernel_av)?;
            }
        }
    }

    Ok(out)
}
