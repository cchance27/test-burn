use metallic_env::{FoundryEnvVar, is_set};

use super::{
    dispatch::{infer_n_kv_heads, prefill_split_k_env, prefill_warps_env}, kernels::{
        FlashDecodeArgs, SdpaPrefillArgs, SdpaPrefillSplitKPartArgs, SdpaPrefillSplitKReduceArgs, get_flash_decode_kernel, get_sdpa_prefill_kernel, get_sdpa_prefill_splitk_part_kernel, get_sdpa_prefill_splitk_reduce_kernel
    }, variants::{FlashDecodeVariant, flash_decode_variant_from_env, select_flash_decode_variant_m2m3}
};
use crate::{
    Foundry, MetalError, metals::{
        common::runtime::require_contiguous_last_dim, flashattention::stages::{SdpaParams, SdpaPrefillSplitKParams}
    }, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F32}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};

#[allow(clippy::too_many_arguments)]
pub fn run_flash_decode_with_variant(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
    variant: FlashDecodeVariant,
) -> Result<(), MetalError> {
    run_flash_decode_impl(
        foundry,
        q,
        k,
        v,
        output,
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        Some(variant),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_flash_decode(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
) -> Result<(), MetalError> {
    run_flash_decode_impl(
        foundry,
        q,
        k,
        v,
        output,
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_flash_decode_impl(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
    variant_override: Option<FlashDecodeVariant>,
) -> Result<(), MetalError> {
    tracing::trace!(
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        has_variant_override = variant_override.is_some(),
        q_dims = ?q.dims(),
        k_dims = ?k.dims(),
        v_dims = ?v.dims(),
        "FlashAttention dispatcher entering"
    );
    if q_seq_len > 1 {
        if variant_override.is_some() {
            return Err(MetalError::OperationNotSupported(
                "run_flash_decode_with_variant is decode-only (q_seq_len must be 1)".into(),
            ));
        }

        if !(head_dim == 64 || head_dim == 128) {
            return Err(MetalError::OperationNotSupported(format!(
                "Prefill only supports head_dim=64 or 128, got {}",
                head_dim
            )));
        }

        let d_model = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| MetalError::OperationNotSupported("d_model overflow".into()))?;

        require_contiguous_last_dim("q", q.strides())?;
        require_contiguous_last_dim("k", k.strides())?;
        require_contiguous_last_dim("v", v.strides())?;
        require_contiguous_last_dim("output", output.strides())?;

        let (q_stride_b, q_stride_h, q_stride_m) = {
            let dims = q.dims();
            let strides = q.strides();
            let q_len = q_seq_len as usize;
            let hd = head_dim as usize;
            let dm = d_model as usize;

            let q_head_major_from_token_meta = if kv_head_major {
                if let [1, _m_cap, dm0] = dims {
                    if *dm0 == dm && strides.len() == 3 {
                        let q_stride_h = q_seq_len
                            .checked_mul(head_dim)
                            .ok_or_else(|| MetalError::OperationNotSupported("Prefill Q stride overflow".into()))?;
                        let q_stride_b = n_heads
                            .checked_mul(q_stride_h)
                            .ok_or_else(|| MetalError::OperationNotSupported("Prefill Q stride overflow".into()))?;
                        Some((q_stride_b, q_stride_h, head_dim))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let strides = if let Some(strides) = q_head_major_from_token_meta {
                strides
            } else {
                match dims {
                    [h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 3 => {
                        (0u32, strides[0] as u32, strides[1] as u32)
                    }
                    [b, h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 4 => {
                        (strides[0] as u32, strides[1] as u32, strides[2] as u32)
                    }
                    [m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 2 => (0u32, head_dim, strides[0] as u32),
                    [b, m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 3 => (strides[0] as u32, head_dim, strides[1] as u32),
                    [m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 3 => {
                        (0u32, strides[1] as u32, strides[0] as u32)
                    }
                    [b, m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 4 => {
                        (strides[0] as u32, strides[2] as u32, strides[1] as u32)
                    }
                    _ => {
                        return Err(MetalError::OperationNotSupported(format!(
                            "Prefill Q layout unsupported: dims={dims:?} strides={strides:?} (expected head-major [H,M>=q_len,D] or token-major [M>=q_len,d_model]/[M>=q_len,H,D])"
                        )));
                    }
                }
            };
            Ok::<(u32, u32, u32), MetalError>(strides)
        }?;

        let (out_stride_b, out_stride_h, out_stride_m) = {
            let dims = output.dims();
            let strides = output.strides();
            let q_len = q_seq_len as usize;
            let hd = head_dim as usize;
            let dm = d_model as usize;

            match dims {
                [m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 2 => (0u32, head_dim, strides[0] as u32),
                [b, m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 3 => (strides[0] as u32, head_dim, strides[1] as u32),
                [m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 3 => {
                    (0u32, strides[1] as u32, strides[0] as u32)
                }
                [b, m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 4 => {
                    (strides[0] as u32, strides[2] as u32, strides[1] as u32)
                }
                [h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 3 => {
                    (0u32, strides[0] as u32, strides[1] as u32)
                }
                [b, h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 4 => {
                    (strides[0] as u32, strides[1] as u32, strides[2] as u32)
                }
                _ => {
                    return Err(MetalError::OperationNotSupported(format!(
                        "Prefill output layout unsupported: dims={dims:?} strides={strides:?}"
                    )));
                }
            }
        };

        let n_kv_heads = infer_n_kv_heads(k, kv_head_major, head_dim);
        let mut group_size = n_heads / n_kv_heads;
        if group_size == 0 {
            group_size = 1;
        }
        let kv_path = if kv_head_major && n_kv_heads < n_heads {
            "compact_gqa"
        } else if kv_head_major {
            "expanded_heads_or_mha"
        } else {
            "token_major"
        };
        tracing::trace!(
            kv_path,
            n_heads,
            n_kv_heads,
            group_size,
            kv_head_major,
            q_seq_len,
            kv_seq_len,
            q_dims = ?q.dims(),
            k_dims = ?k.dims(),
            v_dims = ?v.dims(),
            "FlashAttention prefill KV path resolved"
        );

        if is_set(FoundryEnvVar::DebugSdpa) {
            tracing::info!("SDPA Prefill Debug: Q dims={:?} K dims={:?}", q.dims(), k.dims());
            tracing::info!(
                "SDPA Prefill Debug: n_heads={} n_kv_heads={} group_size={} kv_head_major={}",
                n_heads,
                n_kv_heads,
                group_size,
                kv_head_major
            );
            tracing::info!("SDPA Prefill Debug: K strides={:?}", k.strides());
        }

        let (k_stride_b, k_stride_h, stride_k_s) = {
            let s = k.strides();
            if kv_head_major {
                if s.len() == 4 {
                    (s[0] as u32, s[1] as u32, s[2] as u32)
                } else {
                    (0, s[0] as u32, s[1] as u32)
                }
            } else if s.len() == 4 {
                (s[0] as u32, s[2] as u32, s[1] as u32)
            } else {
                (0, s[1] as u32, s[0] as u32)
            }
        };

        let (v_stride_b, v_stride_h, stride_v_s) = {
            let s = v.strides();
            if kv_head_major {
                if s.len() == 4 {
                    (s[0] as u32, s[1] as u32, s[2] as u32)
                } else {
                    (0, s[0] as u32, s[1] as u32)
                }
            } else if s.len() == 4 {
                (s[0] as u32, s[2] as u32, s[1] as u32)
            } else {
                (0, s[1] as u32, s[0] as u32)
            }
        };

        let prefill_warps = prefill_warps_env().unwrap_or(8);
        if !matches!(prefill_warps, 4 | 8) {
            return Err(MetalError::OperationNotSupported(format!(
                "METALLIC_FA_PREFILL_WARPS must be 4 or 8, got {}",
                prefill_warps
            )));
        }

        let tiling_m = prefill_warps * 4;
        let grid_m_tiles = (q_seq_len + tiling_m - 1) / tiling_m;

        let mut split_k = if is_set(FoundryEnvVar::DisableFaPrefillSplitK) {
            1u32
        } else if let Some(v) = prefill_split_k_env() {
            v.max(1)
        } else if kv_seq_len >= 4096 && q_seq_len >= 16 {
            8
        } else if kv_seq_len >= 2048 && q_seq_len >= 16 {
            4
        } else {
            1
        };
        let kv_tiles = (kv_seq_len + 31) / 32;
        split_k = split_k.min(kv_tiles.max(1));

        let group = ThreadgroupSize::d1((prefill_warps * 32) as usize);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let query_offset = kv_seq_len.saturating_sub(q_seq_len);

        use crate::metals::flashattention::stages::SdpaPrefillParams;
        let args = SdpaPrefillArgs {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            output: TensorArg::from_tensor(output),
            params: SdpaPrefillParams {
                kv_len: kv_seq_len,
                head_dim,
                scale,
                stride_k_s,
                stride_v_s,
                query_offset,
                q_stride_b,
                q_stride_h,
                k_stride_b,
                k_stride_h,
                v_stride_b,
                v_stride_h,
                out_stride_b,
                out_stride_h,
                q_stride_m,
                out_stride_m,
                group_size,
                q_len: q_seq_len,
            },
        };

        if split_k <= 1 {
            let grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, 1);
            let config = DispatchConfig::new(grid, group);
            let kernel = get_sdpa_prefill_kernel(prefill_warps);
            let bound = kernel.clone().bind_arc(args, config);
            return foundry.run(&bound);
        }

        let q_tile_count = grid_m_tiles;
        let tile_m = tiling_m;
        let partial_rows = (split_k as usize)
            .checked_mul(n_heads as usize)
            .and_then(|v| v.checked_mul(q_tile_count as usize))
            .and_then(|v| v.checked_mul(tile_m as usize))
            .ok_or_else(|| MetalError::OperationNotSupported("Split-K scratch size overflow".into()))?;
        let partial_acc_elems = partial_rows
            .checked_mul(head_dim as usize)
            .ok_or_else(|| MetalError::OperationNotSupported("Split-K acc scratch size overflow".into()))?;

        let partial_acc = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_acc_elems], TensorInit::Uninitialized)?;
        let partial_m = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_rows], TensorInit::Uninitialized)?;
        let partial_l = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_rows], TensorInit::Uninitialized)?;

        let splitk_params = SdpaPrefillSplitKParams {
            kv_len: kv_seq_len,
            head_dim,
            scale,
            stride_k_s,
            stride_v_s,
            query_offset,
            q_stride_b,
            q_stride_h,
            k_stride_b,
            k_stride_h,
            v_stride_b,
            v_stride_h,
            out_stride_b,
            out_stride_h,
            q_stride_m,
            out_stride_m,
            group_size,
            q_len: q_seq_len,
            n_heads,
            split_k,
        };

        let part_args = SdpaPrefillSplitKPartArgs {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            partial_acc: TensorArg::from_tensor(&partial_acc),
            partial_m: TensorArg::from_tensor(&partial_m),
            partial_l: TensorArg::from_tensor(&partial_l),
            params: splitk_params,
        };
        let part_grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, split_k as usize);
        let part_config = DispatchConfig::new(part_grid, group);
        let part_kernel = get_sdpa_prefill_splitk_part_kernel(prefill_warps);
        let part_bound = part_kernel.clone().bind_arc(part_args, part_config);
        foundry.run(&part_bound)?;

        let reduce_args = SdpaPrefillSplitKReduceArgs {
            output: TensorArg::from_tensor(output),
            partial_acc: TensorArg::from_tensor(&partial_acc),
            partial_m: TensorArg::from_tensor(&partial_m),
            partial_l: TensorArg::from_tensor(&partial_l),
            params: splitk_params,
        };
        let reduce_grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, 1);
        let reduce_config = DispatchConfig::new(reduce_grid, group);
        let reduce_kernel = get_sdpa_prefill_splitk_reduce_kernel(prefill_warps);
        let reduce_bound = reduce_kernel.clone().bind_arc(reduce_args, reduce_config);
        return foundry.run(&reduce_bound);
    }

    if head_dim != 64 && head_dim != 128 {
        return Err(MetalError::OperationNotSupported(format!(
            "Flash Decode only supports head_dim=64 or 128, got {}",
            head_dim
        )));
    }

    let batch = 1u32;
    let capacity = k.dims().get(1).copied().unwrap_or(kv_seq_len as usize) as u32;
    let q_seq_len_decode = q_seq_len.max(1);
    let (q_stride_b, q_stride_h) = if kv_head_major {
        (n_heads * q_seq_len_decode * head_dim, q_seq_len_decode * head_dim)
    } else {
        (q_seq_len_decode * n_heads * head_dim, head_dim)
    };
    let n_kv_heads = infer_n_kv_heads(k, kv_head_major, head_dim).max(1);
    let cache_heads = if kv_head_major { n_kv_heads } else { n_heads };
    let group_size = (n_heads / n_kv_heads).max(1);
    let kv_path = if kv_head_major && n_kv_heads < n_heads {
        "compact_gqa"
    } else if kv_head_major {
        "expanded_heads_or_mha"
    } else {
        "token_major"
    };
    tracing::trace!(
        kv_path,
        n_heads,
        n_kv_heads,
        cache_heads,
        group_size,
        kv_head_major,
        capacity,
        kv_seq_len,
        q_dims = ?q.dims(),
        k_dims = ?k.dims(),
        v_dims = ?v.dims(),
        "FlashAttention decode KV path resolved"
    );

    let (k_stride_b, k_stride_h) = (cache_heads * capacity * head_dim, capacity * head_dim);
    let (v_stride_b, v_stride_h) = (cache_heads * capacity * head_dim, capacity * head_dim);
    let (out_stride_b, out_stride_h) = (n_heads * head_dim, head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    let sdpa_params = SdpaParams {
        kv_len: kv_seq_len,
        head_dim,
        scale,
        stride_k_s: head_dim,
        stride_v_s: head_dim,
    };

    let args = FlashDecodeArgs {
        q: TensorArg::from_tensor(q),
        k: TensorArg::from_tensor(k),
        v: TensorArg::from_tensor(v),
        output: TensorArg::from_tensor(output),
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_h,
        v_stride_b,
        v_stride_h,
        out_stride_b,
        out_stride_h,
        sdpa_params,
    };

    let variant = if let Some(v) = variant_override {
        v
    } else if let Some(v) = flash_decode_variant_from_env(head_dim)? {
        v
    } else {
        select_flash_decode_variant_m2m3(head_dim, kv_seq_len)
    };
    variant.validate_for_head_dim(head_dim)?;

    let grid = GridSize::new(1, n_heads as usize, batch as usize);
    let group = ThreadgroupSize::d1(variant.threads_per_tg() as usize);
    let config = DispatchConfig::new(grid, group);

    let kernel = get_flash_decode_kernel(head_dim, variant);
    let bound = kernel.clone().bind_arc(args, config);
    foundry.run(&bound)
}
