//! MMA Stage implementations for GEMM compound kernels.
//!
//! These stages wrap the Metal MMA primitives and follow the established
//! Foundry Stage pattern for composition.

use std::sync::Arc;

use metallic_macros::Stage as DeriveStage;
use serde::{Deserialize, Serialize};

use crate::{fusion::MetalPolicy, metals::gemm::step::GemmParams, policy::activation::Activation, types::TensorArg};

// =============================================================================
// TileConfig - Tile size configuration for GEMM
// =============================================================================

/// Tile configuration for GEMM kernels.
/// Different configs optimize for different matrix shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TileConfig {
    /// Default (32, 32, 16) for balanced matrices
    #[default]
    Default,
    /// Skinny M (8, 128, 32) when M is small (short prompts, M=1)
    SkinnyM,
    /// Skinny N (64, 16, 16) when N is small
    SkinnyN,
    /// High Performance (64, 64, 32) for large matrices
    HighPerformance,
    /// Custom tile sizes for experimentation
    Custom { bm: u32, bn: u32, bk: u32, wm: u32, wn: u32 },
}

impl TileConfig {
    /// Select optimal tile config based on matrix dimensions.
    pub fn auto_select(m: usize, n: usize) -> Self {
        if m == 1 || (m <= 16 && n >= 64) {
            TileConfig::SkinnyM
        } else if n <= 16 && m >= 64 {
            TileConfig::SkinnyN
        } else if m >= 64 && n >= 64 {
            // For larger matrices, use 64x64 tiles with BK=32 (better for Q8)
            TileConfig::HighPerformance
        } else {
            TileConfig::Default
        }
    }

    /// Get tile sizes: (BM, BN, BK, WM, WN)
    pub fn tile_sizes(&self) -> (u32, u32, u32, u32, u32) {
        match self {
            TileConfig::Default => (32, 32, 16, 2, 2),
            TileConfig::SkinnyM => (8, 128, 32, 1, 4),
            TileConfig::SkinnyN => (64, 16, 16, 4, 1),
            TileConfig::HighPerformance => (64, 64, 32, 2, 2),
            TileConfig::Custom { bm, bn, bk, wm, wn } => (*bm, *bn, *bk, *wm, *wn),
        }
    }

    /// Total threads per threadgroup for this config.
    pub fn threads_per_tg(&self) -> u32 {
        let (_, _, _, wm, wn) = self.tile_sizes();
        wm * wn * 32 // Each warp is 32 threads
    }

    /// Threadgroup memory padding to avoid bank conflicts.
    pub fn tgp_padding(&self) -> u32 {
        4 // 16 bytes / sizeof(half) = 4 elements padding
    }
}

// =============================================================================
// TileLayoutStage - GEMM tile indexing and configuration
// =============================================================================

/// Stage that sets up tiled layout for GEMM.
/// Emits tile configuration defines and index calculations.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    struct_defs_method = "stage_struct_defs",
    emit = r#"
    // Tile indices with optional swizzling for cache locality
    const int2 tile_idx = BlockSwizzle::swizzle(gid, params.swizzle_log);
    const int tile_m = tile_idx.y;
    const int tile_n = tile_idx.x;
    
    // Early exit if tile is out of bounds
    if (tile_m >= params.tiles_m || tile_n >= params.tiles_n) return;
    
    // Batch offset
    const int batch_idx = gid.z;
    
    // Thread identifiers
    const ushort simd_group_id = lid.x / 32;
    const ushort simd_lane_id = lid.x % 32;
    
    // Threadgroup memory allocation
    threadgroup half As[TGP_MEM_SIZE_A];
    threadgroup half Bs[TGP_MEM_SIZE_B];
    
    // Tile bounds in output
    const short tgp_bm = min(GEMM_BM, params.m - tile_m * GEMM_BM);
    const short tgp_bn = min(GEMM_BN, params.n - tile_n * GEMM_BN);
"#,
    out_var = "void"
)]
pub struct TileLayoutStage {
    #[arg(buffer = 10, metal_type = "constant GemmParams&")]
    pub params: GemmParams,
    #[arg(skip, stage_skip)]
    pub config: TileConfig,
    #[arg(skip, stage_skip)]
    pub transpose_a: bool,
    #[arg(skip, stage_skip)]
    pub transpose_b: bool,
}

impl TileLayoutStage {
    pub fn new(config: TileConfig, transpose_a: bool, transpose_b: bool) -> Self {
        Self {
            params: GemmParams::default(),
            config,
            transpose_a,
            transpose_b,
        }
    }
    fn stage_struct_defs(&self) -> String {
        let (bm, bn, bk, wm, wn) = self.config.tile_sizes();
        let tgp_padding = self.config.tgp_padding();
        let tgp_size = self.config.threads_per_tg();

        // Include GemmParams struct definition from MetalStruct derive
        let gemm_params_def = GemmParams::METAL_STRUCT_DEF;

        // BlockSwizzle from MLX for cache-friendly tile indexing
        let block_swizzle = r#"
struct BlockSwizzle {
    static METAL_FUNC int2 swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
        const int tid_x = (tid.x) >> swizzle_log;
        const int tid_y = ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
        return int2(tid_x, tid_y);
    }
};
"#;

        format!(
            r#"
{gemm_params_def}

{block_swizzle}

// GEMM Tile Configuration
// Use GEMM_ prefix to avoid conflicts with template parameters
#define GEMM_BM {bm}
#define GEMM_BN {bn}
#define GEMM_BK {bk}
#define GEMM_WM {wm}
#define GEMM_WN {wn}
#define GEMM_TGP_SIZE {tgp_size}
#define GEMM_TGP_PADDING {tgp_padding}
#define GEMM_TRANSPOSE_A {ta}
#define GEMM_TRANSPOSE_B {tb}

// Threadgroup memory sizes
#define TGP_MEM_SIZE_A (GEMM_TRANSPOSE_A ? GEMM_BK * (GEMM_BM + GEMM_TGP_PADDING) : GEMM_BM * (GEMM_BK + GEMM_TGP_PADDING))
#define TGP_MEM_SIZE_B (GEMM_TRANSPOSE_B ? GEMM_BN * (GEMM_BK + GEMM_TGP_PADDING) : GEMM_BK * (GEMM_BN + GEMM_TGP_PADDING))
"#,
            gemm_params_def = gemm_params_def,
            block_swizzle = block_swizzle,
            bm = bm,
            bn = bn,
            bk = bk,
            wm = wm,
            wn = wn,
            tgp_size = tgp_size,
            tgp_padding = tgp_padding,
            ta = self.transpose_a as u32,
            tb = self.transpose_b as u32
        )
    }
}

// =============================================================================
// TileLoadAStage - Load matrix A tile with Policy
// =============================================================================

/// Stage that loads matrix A (activations) into threadgroup memory.
/// Uses SimpleTileLoader for F16 (no dequant needed).
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("mma/tile_loader.metal"),
    policy_field = "policy",
    template_bindings(
        batch_offset = "self.policy.bytes(\"batch_idx * params.batch_stride_a\")",
        tile_offset = "self.policy.bytes(if self.transpose_a { \"tile_m * GEMM_BM\" } else { \"tile_m * GEMM_BM * params.lda\" })"
    ),
    emit = r#"
    // Initialize A tile loader (F16 activations)
    const device uchar* A_batch = (const device uchar*)a + {batch_offset};
    const device uchar* A_tile = A_batch + {tile_offset};
    
    SimpleTileLoader<half, 
                     GEMM_TRANSPOSE_A ? GEMM_BK : GEMM_BM, 
                     GEMM_TRANSPOSE_A ? GEMM_BM : GEMM_BK,
                     GEMM_TRANSPOSE_A ? GEMM_BM + GEMM_TGP_PADDING : GEMM_BK + GEMM_TGP_PADDING,
                     !GEMM_TRANSPOSE_A,
                     GEMM_TGP_SIZE> loader_a(
        (const device half*)A_tile, params.lda, As, simd_group_id, simd_lane_id
    );
"#,
    out_var = "loader_a"
)]
pub struct TileLoadAStage {
    #[arg(buffer = 0, metal_type = "const device half*")]
    pub a: TensorArg,
    /// Policy for A (typically F16 for activations)
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
    /// Whether A is transposed
    #[arg(skip, stage_skip)]
    pub transpose_a: bool,
}

impl TileLoadAStage {
    pub fn new(policy: Arc<dyn MetalPolicy>, transpose_a: bool) -> Self {
        Self {
            a: TensorArg::default(),
            policy,
            transpose_a,
        }
    }
}

// =============================================================================
// TileLoadBStage - Load matrix B tile with Policy (for quantized weights)
// =============================================================================

/// Stage that loads matrix B (weights) into threadgroup memory.
/// Uses Policy templates for transparent F16/Q8/Q4 dequantization.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    policy_field = "policy",
    template_bindings(batch_offset = "self.policy.bytes(\"batch_idx * params.batch_stride_b\")"),
    emit = r#"
    // Initialize B tile loader with {policy_struct} dequantization
    // Use batch-base and tile-offset for correct global scale indexing
    const device uchar* B_batch = b + {batch_offset};
    const uint blocks_per_k = (params.k + weights_per_block - 1) / weights_per_block;
    
    TileLoader<{policy_struct}, half,
               GEMM_TRANSPOSE_B ? GEMM_BN : GEMM_BK,
               GEMM_TRANSPOSE_B ? GEMM_BK : GEMM_BN,
               GEMM_TRANSPOSE_B ? GEMM_BK + GEMM_TGP_PADDING : GEMM_BN + GEMM_TGP_PADDING,
               GEMM_TRANSPOSE_B,
               GEMM_TGP_SIZE> loader_b(
        B_batch, params.ldb, Bs,
        b_scales, weights_per_block,
        b_is_canonical,
        tile_n * GEMM_BN,  // row_idx_offset
        blocks_per_k,
        params.n,
        simd_group_id, simd_lane_id
    );
"#,
    out_var = "loader_b"
)]
pub struct TileLoadBStage {
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub b: TensorArg,
    #[arg(buffer = 5, metal_type = "const device uchar*")]
    pub b_scales: TensorArg,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    pub weights_per_block: u32,
    #[arg(buffer = 9, metal_type = "constant uint&")]
    pub b_is_canonical: u32,
    /// Policy for B (F16, Q8, etc.)
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
    /// Whether B is transposed
    #[arg(skip, stage_skip)]
    pub transpose_b: bool,
}

impl TileLoadBStage {
    pub fn new(policy: Arc<dyn MetalPolicy>, transpose_b: bool) -> Self {
        Self {
            b: TensorArg::default(),
            b_scales: TensorArg::default(),
            weights_per_block: 0,
            b_is_canonical: 0,
            policy,
            transpose_b,
        }
    }
}

// =============================================================================
// MmaLoopStage - Main GEMM compute loop (Policy-agnostic)
// =============================================================================

/// Stage that performs the tiled GEMM computation.
/// Uses SimdgroupMma for hardware-accelerated matrix multiply.
///
/// This stage is POLICY-AGNOSTIC - it operates on already-dequantized
/// tiles in threadgroup memory.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("mma/mma.metal", "policies/activations.metal"),
    template_bindings(k_remainder = "if !self.k_aligned { r#\"
    // Handle K remainder
    if (params.gemm_k_remainder > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        short2 tile_dims_a = GEMM_TRANSPOSE_A 
            ? short2(tgp_bm, params.gemm_k_remainder) 
            : short2(params.gemm_k_remainder, tgp_bm);
        short2 tile_dims_b = GEMM_TRANSPOSE_B 
            ? short2(params.gemm_k_remainder, tgp_bn) 
            : short2(tgp_bn, params.gemm_k_remainder);
        
        loader_a.load_safe(tile_dims_a);
        loader_b.load_safe(tile_dims_b);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
    }
\"# } else { \"\" }"),
    emit = r#"
    // Initialize simdgroup MMA operator
    SimdgroupMma<half, half, GEMM_BM, GEMM_BN, GEMM_BK, GEMM_WM, GEMM_WN,
                 GEMM_TRANSPOSE_A, GEMM_TRANSPOSE_B,
                 GEMM_TRANSPOSE_A ? GEMM_BM + GEMM_TGP_PADDING : GEMM_BK + GEMM_TGP_PADDING,
                 GEMM_TRANSPOSE_B ? GEMM_BK + GEMM_TGP_PADDING : GEMM_BN + GEMM_TGP_PADDING> mma_op(
        simd_group_id, simd_lane_id
    );
    
    // Main GEMM loop over K dimension
    for (int k = 0; k < params.gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load A and B tiles to threadgroup memory
        if (tgp_bm == GEMM_BM && tgp_bn == GEMM_BN) {
            loader_a.load_unsafe();
            loader_b.load_unsafe();
        } else {
            loader_a.load_safe(GEMM_TRANSPOSE_A ? short2(tgp_bm, GEMM_BK) : short2(GEMM_BK, tgp_bm));
            loader_b.load_safe(GEMM_TRANSPOSE_B ? short2(GEMM_BK, tgp_bn) : short2(tgp_bn, GEMM_BK));
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Multiply-accumulate
        mma_op.mma(As, Bs);
        
        // Advance to next K tile
        loader_a.next();
        loader_b.next();
    }
    {k_remainder}
"#,
    out_var = "mma_op"
)]
pub struct MmaLoopStage {
    /// Whether K is aligned to BK
    #[arg(skip, stage_skip)]
    pub k_aligned: bool,
}

impl MmaLoopStage {
    pub fn new() -> Self {
        Self { k_aligned: true }
    }

    pub fn with_k_aligned(mut self, aligned: bool) -> Self {
        self.k_aligned = aligned;
        self
    }
}

impl Default for MmaLoopStage {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GemmEpilogueStage - Alpha/beta scaling and output write
// =============================================================================

/// Stage that applies epilogue (alpha*result + beta*C) and writes output.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    activation_field = "activation",
    template_bindings(
        has_alpha_beta = "if self.has_alpha_beta { \"true\" } else { \"false\" }",
        has_bias = "if self.has_bias { \"true\" } else { \"false\" }",
        has_activation = "if self.activation != Activation::None { \"true\" } else { \"false\" }"
    ),
    emit = r#"
    if ({has_alpha_beta}) {
        // Apply alpha/beta epilogue
        const device half* C_tile = c + batch_idx * params.batch_stride_c
                                   + tile_m * GEMM_BM * params.ldc + tile_n * GEMM_BN;
        if (tgp_bm == GEMM_BM && tgp_bn == GEMM_BN) {
            mma_op.apply_epilogue(C_tile, params.ldc, alpha, beta);
        } else {
            mma_op.apply_epilogue_safe(C_tile, params.ldc, alpha, beta, short2(tgp_bn, tgp_bm));
        }
    }

    if ({has_bias}) {
        // Apply bias
        mma_op.apply_bias(bias, tile_n * GEMM_BN);
    }

    if ({has_activation}) {
        // Apply activation
        mma_op.apply_activation<{activation_struct}>();
    }

    // Write output
    device half* D_tile = d + batch_idx * params.batch_stride_d
                         + tile_m * GEMM_BM * params.ldd + tile_n * GEMM_BN;
    
    // Use safe store for edge tiles
    if (tgp_bm == GEMM_BM && tgp_bn == GEMM_BN) {
        mma_op.store_result(D_tile, params.ldd);
    } else {
        mma_op.store_result_safe(D_tile, params.ldd, short2(tgp_bn, tgp_bm));
    }
"#,
    out_var = "void"
)]
pub struct GemmEpilogueStage {
    #[arg(buffer = 2, output)]
    pub d: TensorArg,
    #[arg(buffer = 3, metal_type = "const device half*")]
    pub c: TensorArg,
    #[arg(buffer = 7, metal_type = "constant float&")]
    pub alpha: f32,
    #[arg(buffer = 8, metal_type = "constant float&")]
    pub beta: f32,
    #[arg(buffer = 4, metal_type = "const device half*")]
    pub bias: TensorArg,
    /// Whether to apply alpha/beta scaling
    #[arg(skip, stage_skip)]
    pub has_alpha_beta: bool,
    /// Whether to apply bias
    #[arg(skip, stage_skip)]
    pub has_bias: bool,
    /// Activation to apply
    #[arg(skip, stage_skip)]
    pub activation: Activation,
}

impl GemmEpilogueStage {
    pub fn new() -> Self {
        Self {
            d: TensorArg::default(),
            c: TensorArg::default(),
            alpha: 1.0,
            beta: 0.0,
            bias: TensorArg::default(),
            has_alpha_beta: false,
            has_bias: false,
            activation: Activation::None,
        }
    }

    pub fn with_alpha_beta(mut self) -> Self {
        self.has_alpha_beta = true;
        self
    }

    pub fn with_bias(mut self) -> Self {
        self.has_bias = true;
        self
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Default for GemmEpilogueStage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_auto_select() {
        assert_eq!(TileConfig::auto_select(1, 4096), TileConfig::SkinnyM);
        assert_eq!(TileConfig::auto_select(512, 4096), TileConfig::HighPerformance);
        assert_eq!(TileConfig::auto_select(512, 8), TileConfig::SkinnyN);
    }

    #[test]
    fn test_tile_config_sizes() {
        let (bm, bn, bk, wm, wn) = TileConfig::Default.tile_sizes();
        assert_eq!((bm, bn, bk, wm, wn), (32, 32, 16, 2, 2));

        let tgp = TileConfig::Default.threads_per_tg();
        assert_eq!(tgp, 128); // 2*2*32
    }
}
