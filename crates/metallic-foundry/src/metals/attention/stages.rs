use crate::compound::{BufferArg, Stage};

/// KV tiling configuration shared by attention kernels.
///
/// The intent is to standardize a "block N" (sequence-length) tile size so
/// FlashAttention and future attention kernels can share loader code and tuning knobs.
#[derive(Clone, Copy, Debug)]
pub struct KvTileConfig {
    /// Tile width in KV positions (N dimension).
    pub bn: u32,
}

impl KvTileConfig {
    pub const fn new(bn: u32) -> Self {
        Self { bn }
    }
}

/// Stage that defines tile constants and computes the KV block range for this threadgroup.
///
/// This stage is intentionally lightweight: it does not perform loads itself, but sets up
/// the indices and compile-time constants that loader stages can rely on.
#[derive(Clone, Debug)]
pub struct KvTileLayoutStage {
    pub config: KvTileConfig,
}

impl KvTileLayoutStage {
    pub fn new(config: KvTileConfig) -> Self {
        Self { config }
    }
}

impl Stage for KvTileLayoutStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // This stage references `sdpa_params` in its emit block, so it assumes the parent kernel
        // binds `sdpa_params` somewhere. We don't bind it here to avoid forcing a specific layout.
        vec![]
    }

    fn struct_defs(&self) -> String {
        // Provide the tile constant as a `#define`.
        // Defensive: keep this stage self-contained even if multiple kernels include it.
        format!(
            r#"
#ifndef METALLIC_KV_TILE_BN
#define METALLIC_KV_TILE_BN
#define KV_TILE_BN {}
#endif
"#,
            self.config.bn.max(1)
        )
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        (
            "kv_tile".to_string(),
            r#"
    // KV tiling (block-N) configuration
    const uint kv_bn = KV_TILE_BN;

    // In Foundry compound kernels, `gid` is `[[threadgroup_position_in_grid]]`.
    // Convention for attention kernels: gid.x = kv_block, gid.y = head, gid.z = batch.
    const uint kv_block = gid.x;

    const uint kv_start = kv_block * kv_bn;
    const uint kv_end = min(kv_start + kv_bn, sdpa_params.kv_len);
"#
            .to_string(),
        )
    }
}
