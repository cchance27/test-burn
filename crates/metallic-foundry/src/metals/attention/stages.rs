use metallic_macros::Stage;

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
#[derive(Clone, Debug, Stage)]
#[stage(
    template_bindings(kv_tile_bn = "self.config.bn.max(1)"),
    struct_defs_method = "stage_struct_defs",
    emit = r#"
    // KV tiling (block-N) configuration
    const uint kv_bn = {kv_tile_bn};

    // In Foundry compound kernels, `gid` is `[[threadgroup_position_in_grid]]`.
    // Convention for attention kernels: gid.x = kv_block, gid.y = head, gid.z = batch.
    const uint kv_block = gid.x;

    const uint kv_start = kv_block * kv_bn;
    const uint kv_end = min(kv_start + kv_bn, sdpa_params.kv_len);
"#,
    out_var = "kv_tile"
)]
pub struct KvTileLayoutStage {
    #[arg(skip, stage_skip)]
    pub config: KvTileConfig,
}

impl KvTileLayoutStage {
    pub fn new(config: KvTileConfig) -> Self {
        Self { config }
    }

    fn stage_struct_defs(&self) -> String {
        format!("#define KV_TILE_BN {}", self.config.bn.max(1))
    }
}
