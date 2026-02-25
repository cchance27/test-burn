use std::sync::OnceLock;

use metallic_env::{FA_PREFILL_ENGINE, FA_PREFILL_SPLIT_K, FA_PREFILL_WARPS};

use crate::{MetalError, metals::flashattention::variants::FlashPrefillEngine, types::TensorArg};

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
pub(super) fn prefill_engine_env() -> Result<Option<FlashPrefillEngine>, MetalError> {
    FA_PREFILL_ENGINE
        .get_valid()
        .map(|v| match v.trim().to_ascii_lowercase().as_str() {
            "fa1" | "baseline" => Ok(FlashPrefillEngine::Fa1),
            "fa2" | "fa2_pipelined" | "pipelined" => Ok(FlashPrefillEngine::Fa2Pipelined),
            other => Err(MetalError::OperationNotSupported(format!(
                "Invalid METALLIC_FA_PREFILL_ENGINE='{other}'; expected one of: fa1, fa2"
            ))),
        })
        .transpose()
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
fn classify_prefill_device(working_set_bytes: u64) -> PrefillDeviceClass {
    const GIB: u64 = 1024 * 1024 * 1024;
    if working_set_bytes < 12 * GIB {
        PrefillDeviceClass::Low
    } else if working_set_bytes < 24 * GIB {
        PrefillDeviceClass::Standard
    } else {
        PrefillDeviceClass::High
    }
}

#[inline]
pub(super) fn select_prefill_warps(storage_bytes: usize, kv_seq_len: u32, q_seq_len: u32, working_set_bytes: u64) -> u32 {
    let base = if storage_bytes <= 2 { 8 } else { 4 };
    match classify_prefill_device(working_set_bytes) {
        PrefillDeviceClass::Low => {
            if kv_seq_len >= 4096 || q_seq_len >= 64 {
                4
            } else {
                base
            }
        }
        PrefillDeviceClass::Standard | PrefillDeviceClass::High => base,
    }
}

#[inline]
pub(super) fn select_prefill_split_k(kv_seq_len: u32, q_seq_len: u32, storage_bytes: usize, working_set_bytes: u64) -> u32 {
    let base = if kv_seq_len >= 4096 && q_seq_len >= 16 {
        8
    } else if kv_seq_len >= 2048 && q_seq_len >= 16 {
        4
    } else {
        1
    };
    let base = if storage_bytes <= 2 { base } else { base.min(4) };
    match classify_prefill_device(working_set_bytes) {
        PrefillDeviceClass::Low => {
            if q_seq_len <= 16 {
                base.min(4)
            } else {
                base
            }
        }
        PrefillDeviceClass::Standard | PrefillDeviceClass::High => base,
    }
}

#[inline]
pub(super) fn select_prefill_engine(storage_bytes: usize, kv_seq_len: u32, q_seq_len: u32, working_set_bytes: u64) -> FlashPrefillEngine {
    if storage_bytes > 2 {
        return FlashPrefillEngine::Fa1;
    }

    match classify_prefill_device(working_set_bytes) {
        PrefillDeviceClass::Low => {
            if kv_seq_len >= 512 && q_seq_len >= 8 {
                FlashPrefillEngine::Fa2Pipelined
            } else {
                FlashPrefillEngine::Fa1
            }
        }
        PrefillDeviceClass::Standard | PrefillDeviceClass::High => {
            if kv_seq_len >= 128 || q_seq_len >= 8 {
                FlashPrefillEngine::Fa2Pipelined
            } else {
                FlashPrefillEngine::Fa1
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillDeviceClass {
    Low,
    Standard,
    High,
}

#[cfg(test)]
mod tests {
    use metallic_env::{EnvVarGuard, FoundryEnvVar};
    use serial_test::serial;

    use super::{prefill_engine_env, select_prefill_engine, select_prefill_split_k, select_prefill_warps};
    use crate::metals::flashattention::variants::FlashPrefillEngine;

    #[test]
    fn prefill_warps_keeps_f16_fast_path() {
        assert_eq!(select_prefill_warps(2, 1024, 8, 24 * 1024 * 1024 * 1024), 8);
    }

    #[test]
    fn prefill_warps_scales_down_for_wider_storage() {
        assert_eq!(select_prefill_warps(4, 1024, 8, 24 * 1024 * 1024 * 1024), 4);
    }

    #[test]
    fn prefill_splitk_keeps_existing_f16_thresholds() {
        assert_eq!(select_prefill_split_k(4096, 16, 2, 24 * 1024 * 1024 * 1024), 8);
        assert_eq!(select_prefill_split_k(2048, 16, 2, 24 * 1024 * 1024 * 1024), 4);
        assert_eq!(select_prefill_split_k(1024, 8, 2, 24 * 1024 * 1024 * 1024), 1);
    }

    #[test]
    fn prefill_splitk_caps_wider_storage() {
        assert_eq!(select_prefill_split_k(4096, 16, 4, 24 * 1024 * 1024 * 1024), 4);
    }

    #[test]
    fn prefill_selector_uses_lighter_warps_on_low_memory_large_shapes() {
        assert_eq!(select_prefill_warps(2, 4096, 64, 8 * 1024 * 1024 * 1024), 4);
    }

    #[test]
    fn prefill_splitk_low_memory_prefers_lower_split_for_short_prefill() {
        assert_eq!(select_prefill_split_k(4096, 16, 2, 8 * 1024 * 1024 * 1024), 4);
    }

    #[test]
    fn prefill_engine_prefers_fa2_on_f16_medium_to_large_shapes() {
        let engine = select_prefill_engine(2, 1024, 16, 24 * 1024 * 1024 * 1024);
        assert_eq!(engine, FlashPrefillEngine::Fa2Pipelined);
    }

    #[test]
    fn prefill_engine_keeps_fa1_for_wider_storage() {
        let engine = select_prefill_engine(4, 4096, 32, 24 * 1024 * 1024 * 1024);
        assert_eq!(engine, FlashPrefillEngine::Fa1);
    }

    #[test]
    #[serial]
    fn prefill_engine_override_parses_valid_value() {
        let _guard = EnvVarGuard::set(FoundryEnvVar::FaPrefillEngine, "fa2");
        let parsed = prefill_engine_env().expect("override parse");
        assert_eq!(parsed, Some(FlashPrefillEngine::Fa2Pipelined));
    }

    #[test]
    #[serial]
    fn prefill_engine_override_rejects_invalid_value() {
        let _guard = EnvVarGuard::set(FoundryEnvVar::FaPrefillEngine, "nope");
        let err = prefill_engine_env().expect_err("invalid override should fail-fast");
        let msg = format!("{err}");
        assert!(msg.contains("Invalid METALLIC_FA_PREFILL_ENGINE"), "unexpected error: {msg}");
    }
}
