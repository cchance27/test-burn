use metallic_env::{FA_PREFILL_SPLIT_K, FA_PREFILL_WARPS};

use crate::types::TensorArg;

#[inline]
pub(super) fn prefill_warps_env() -> Option<u32> {
    FA_PREFILL_WARPS.get().ok().flatten()
}

#[inline]
pub(super) fn prefill_split_k_env() -> Option<u32> {
    FA_PREFILL_SPLIT_K.get().ok().flatten()
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
