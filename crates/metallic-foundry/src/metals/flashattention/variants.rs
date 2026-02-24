use std::sync::OnceLock;

use metallic_env::{FA_DECODE_KEYS_PER_WARP, FA_DECODE_SCALAR, FA_DECODE_TG_OUT, FA_DECODE_WARPS};

use crate::MetalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashDecodeScalar {
    Half2,
    Half4,
}

impl FlashDecodeScalar {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Half2 => "half2",
            Self::Half4 => "half4",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashDecodeTgOut {
    Float,
    Half,
}

impl FlashDecodeTgOut {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float => "tgf",
            Self::Half => "tgh",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashDecodeVariant {
    pub warps: u32,
    pub keys_per_warp: u32,
    pub scalar: FlashDecodeScalar,
    pub tg_out: FlashDecodeTgOut,
}

impl FlashDecodeVariant {
    pub const fn threads_per_tg(self) -> u32 {
        self.warps * 32
    }

    pub const fn keys_per_block(self) -> u32 {
        self.warps * self.keys_per_warp
    }

    pub fn cache_key_suffix(self) -> String {
        format!(
            "w{}_k{}_{}_{}",
            self.warps,
            self.keys_per_warp,
            self.scalar.as_str(),
            self.tg_out.as_str()
        )
    }

    pub fn validate_for_head_dim(self, head_dim: u32) -> Result<(), MetalError> {
        if !(1..=32).contains(&self.warps) {
            return Err(MetalError::OperationNotSupported(format!(
                "FlashDecodeVariant invalid warps: {}",
                self.warps
            )));
        }
        if !(1..=64).contains(&self.keys_per_warp) {
            return Err(MetalError::OperationNotSupported(format!(
                "FlashDecodeVariant invalid keys_per_warp: {}",
                self.keys_per_warp
            )));
        }

        let threads = self.threads_per_tg();
        if !matches!(threads, 128 | 256 | 384 | 512) {
            return Err(MetalError::OperationNotSupported(format!(
                "FlashDecodeVariant unsupported threads_per_tg: {} (warps={})",
                threads, self.warps
            )));
        }

        if head_dim == 64 && matches!(self.scalar, FlashDecodeScalar::Half2 | FlashDecodeScalar::Half4) {
            return Ok(());
        }

        if matches!(self.scalar, FlashDecodeScalar::Half2) {
            return Err(MetalError::OperationNotSupported(format!(
                "FlashDecodeVariant Half2 is only supported for head_dim=64 (got head_dim={head_dim})"
            )));
        }

        if head_dim == 0 || head_dim > 128 || !head_dim.is_multiple_of(4) {
            return Err(MetalError::OperationNotSupported(format!(
                "FlashDecodeVariant Half4 requires head_dim to be a non-zero multiple of 4 and <= 128 (got head_dim={head_dim})"
            )));
        }

        Ok(())
    }
}

fn parse_env_u32(key: &'static str) -> Option<u32> {
    match key {
        "METALLIC_FA_DECODE_WARPS" => FA_DECODE_WARPS.get().ok().flatten(),
        "METALLIC_FA_DECODE_KEYS_PER_WARP" => FA_DECODE_KEYS_PER_WARP.get().ok().flatten(),
        _ => None,
    }
}

fn parse_env_scalar() -> Option<FlashDecodeScalar> {
    FA_DECODE_SCALAR
        .get()
        .ok()
        .flatten()
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "half2" => Some(FlashDecodeScalar::Half2),
            "half4" => Some(FlashDecodeScalar::Half4),
            _ => None,
        })
}

fn parse_env_tg_out() -> Option<FlashDecodeTgOut> {
    FA_DECODE_TG_OUT
        .get()
        .ok()
        .flatten()
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "float" | "f32" | "tgf" => Some(FlashDecodeTgOut::Float),
            "half" | "f16" | "tgh" => Some(FlashDecodeTgOut::Half),
            _ => None,
        })
}

#[derive(Clone, Copy, Debug)]
struct EnvOverride {
    warps: Option<u32>,
    keys_per_warp: Option<u32>,
    scalar: Option<FlashDecodeScalar>,
    tg_out: Option<FlashDecodeTgOut>,
}

fn decode_env_override() -> &'static EnvOverride {
    static OVERRIDE: OnceLock<EnvOverride> = OnceLock::new();
    OVERRIDE.get_or_init(|| EnvOverride {
        warps: parse_env_u32("METALLIC_FA_DECODE_WARPS"),
        keys_per_warp: parse_env_u32("METALLIC_FA_DECODE_KEYS_PER_WARP"),
        scalar: parse_env_scalar(),
        tg_out: parse_env_tg_out(),
    })
}

pub fn flash_decode_variant_from_env(head_dim: u32) -> Result<Option<FlashDecodeVariant>, MetalError> {
    let ov = decode_env_override();
    if ov.warps.is_none() && ov.keys_per_warp.is_none() && ov.scalar.is_none() && ov.tg_out.is_none() {
        return Ok(None);
    }

    // Default scalar selection if only warps/keys were provided.
    let default_scalar = if head_dim == 64 {
        FlashDecodeScalar::Half2
    } else {
        FlashDecodeScalar::Half4
    };

    let variant = FlashDecodeVariant {
        warps: ov.warps.unwrap_or(if head_dim == 64 { 16 } else { 8 }),
        keys_per_warp: ov.keys_per_warp.unwrap_or(if head_dim == 64 { 32 } else { 16 }),
        scalar: ov.scalar.unwrap_or(default_scalar),
        tg_out: ov.tg_out.unwrap_or(FlashDecodeTgOut::Float),
    };

    variant.validate_for_head_dim(head_dim)?;
    Ok(Some(variant))
}

/// Default decode variant selector tuned for Apple M2/M3.
///
/// This is intentionally conservative and only selects from a small table of known-good
/// configurations used by supported decode head dims.
pub fn select_flash_decode_variant_m2m3(head_dim: u32, kv_len: u32) -> FlashDecodeVariant {
    match head_dim {
        64 => {
            // In end-to-end throughput runs on M2/M3, Half2 consistently wins for D=64 even
            // when Half4 looks competitive in isolated microbenches.
            if kv_len >= 256 {
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 32,
                    scalar: FlashDecodeScalar::Half2,
                    tg_out: FlashDecodeTgOut::Float,
                }
            } else {
                FlashDecodeVariant {
                    warps: 8,
                    keys_per_warp: 16,
                    scalar: FlashDecodeScalar::Half2,
                    tg_out: FlashDecodeTgOut::Float,
                }
            }
        }
        128 => {
            // Fused RoPE→decode sweep shows a large win from using 16 warps for D=128.
            // For very long KV, storing warp partials in half can reduce threadgroup bandwidth.
            let tg_out = if kv_len >= 2048 {
                FlashDecodeTgOut::Half
            } else {
                FlashDecodeTgOut::Float
            };
            if kv_len >= 192 {
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 16,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out,
                }
            } else {
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 8,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out,
                }
            }
        }
        _ => FlashDecodeVariant {
            warps: 8,
            keys_per_warp: 16,
            scalar: FlashDecodeScalar::Half4,
            tg_out: FlashDecodeTgOut::Float,
        },
    }
}

/// Storage-byte aware selector entrypoint.
///
/// Current FA kernels are F16-only and this still routes to the tuned M2/M3 table.
/// The storage-byte input is intentionally kept for future fp32/bf16 FA expansion.
pub fn select_flash_decode_variant(head_dim: u32, kv_len: u32, _storage_bytes: usize) -> FlashDecodeVariant {
    select_flash_decode_variant_m2m3(head_dim, kv_len)
}

#[cfg(test)]
mod tests {
    use super::{select_flash_decode_variant, select_flash_decode_variant_m2m3};

    #[test]
    fn storage_aware_selector_matches_tuned_table_for_current_paths() {
        let tuned = select_flash_decode_variant_m2m3(64, 1024);
        let aware = select_flash_decode_variant(64, 1024, 2);
        assert_eq!(aware, tuned);
    }

    #[test]
    fn storage_aware_selector_accepts_wider_storage_for_future_paths() {
        let a = select_flash_decode_variant(128, 4096, 4);
        let b = select_flash_decode_variant_m2m3(128, 4096);
        assert_eq!(a, b);
    }
}
