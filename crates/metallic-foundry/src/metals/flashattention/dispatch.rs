use std::sync::OnceLock;

use metallic_env::{FA_PREFILL_SPLIT_K, FA_PREFILL_WARPS};

use crate::types::TensorArg;

#[inline]
pub(super) fn prefill_warps_env() -> Option<u32> {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    FA_PREFILL_WARPS.get_valid_cached(&OVERRIDE)
}

#[inline]
pub(super) fn prefill_split_k_env() -> Option<u32> {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    FA_PREFILL_SPLIT_K.get_valid_cached(&OVERRIDE)
}

#[inline]
pub(super) fn infer_n_kv_heads(k: &TensorArg, kv_head_major: bool, head_dim: u32) -> u32 {
    if kv_head_major {
        // Head-major:
        // Rank-3: [Heads, Seq, Dim] -> dims[0]
        // Rank-4: [Batch, Heads, Seq, Dim] -> dims[1]
        let dims = k.dims();
        if dims.len() == 3 {
            dims[0] as u32
        } else {
            dims.get(1).copied().unwrap_or(1) as u32
        }
    } else {
        // Token-major: [Batch, Seq, Heads*Dim] (or rank-3 equivalent)
        let last = k.dims().last().copied().unwrap_or(head_dim as usize) as u32;
        (last / head_dim).max(1)
    }
}

#[inline]
pub(super) fn select_prefill_warps(storage_bytes: usize) -> u32 {
    if storage_bytes <= 2 { 8 } else { 4 }
}

#[inline]
pub(super) fn select_prefill_split_k(kv_seq_len: u32, q_seq_len: u32, storage_bytes: usize) -> u32 {
    let base = if kv_seq_len >= 4096 && q_seq_len >= 16 {
        8
    } else if kv_seq_len >= 2048 && q_seq_len >= 16 {
        4
    } else {
        1
    };
    if storage_bytes <= 2 { base } else { base.min(4) }
}

#[cfg(test)]
mod tests {
    use super::{select_prefill_split_k, select_prefill_warps};

    #[test]
    fn prefill_warps_keeps_f16_fast_path() {
        assert_eq!(select_prefill_warps(2), 8);
    }

    #[test]
    fn prefill_warps_scales_down_for_wider_storage() {
        assert_eq!(select_prefill_warps(4), 4);
    }

    #[test]
    fn prefill_splitk_keeps_existing_f16_thresholds() {
        assert_eq!(select_prefill_split_k(4096, 16, 2), 8);
        assert_eq!(select_prefill_split_k(2048, 16, 2), 4);
        assert_eq!(select_prefill_split_k(1024, 8, 2), 1);
    }

    #[test]
    fn prefill_splitk_caps_wider_storage() {
        assert_eq!(select_prefill_split_k(4096, 16, 4), 4);
    }
}
