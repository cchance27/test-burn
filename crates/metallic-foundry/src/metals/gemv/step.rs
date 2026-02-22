//! GemvV2Step - Full-featured GEMV using Stage composition.
//!
//! Features:
//! - Canonical 4x unrolling (matching legacy performance)
//! - NK/KN layout support via LayoutStage
//! - Policy templates for Q8/F16 transparency
//! - Dynamic block size selection based on K dimension
//! - Composable stage architecture

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::stages::{CanonicalDotStage, ScalarDotStage, VectorizedDotStage, WarpWriteOutputStage};
use crate::{
    compound::{
        CompiledCompoundKernel, CompoundKernel, Layout, stages::{ThreadLayoutStage, WarpLayoutStage, WarpReduceStage}
    }, policy::activation::Activation, types::TensorArg
};

fn use_f16_cols8() -> bool {
    // Default ON: this path mirrors the legacy Context RowMajor FP16 GEMV pointer arithmetic and is consistently faster
    // for decode-heavy shapes (e.g. K=896, K=4864). Allow an escape hatch to disable for debugging/regressions.
    std::env::var("METALLIC_GEMV_F16_COLS8").ok().map(|val| val != "0").unwrap_or(true)
}

// =============================================================================
// Parameters and Arguments
// =============================================================================

/// Arguments for GemvV2 kernel dispatch.
#[derive(Debug, KernelArgs)]
pub struct GemvV2Args {
    #[arg(buffer = 0)]
    pub weights: TensorArg,
    #[arg(buffer = 1)]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 2)]
    pub input: TensorArg,
    #[arg(buffer = 3, output)]
    pub output: TensorArg,
    #[arg(buffer = 4)]
    pub k_dim: u32,
    #[arg(buffer = 5)]
    pub n_dim: u32,
    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    #[arg(buffer = 7)]
    pub bias: TensorArg,
    #[arg(buffer = 8)]
    pub has_bias: u32,
    #[arg(buffer = 9)]
    pub alpha: f32,
    #[arg(buffer = 10)]
    pub residual: TensorArg,
    #[arg(buffer = 11)]
    pub has_residual: u32,
    #[arg(buffer = 12)]
    pub beta: f32,
}

// =============================================================================
// Fast Warp-Per-Row Kernels (Standard V2)
// =============================================================================

/// Sub-strategy for GEMV V2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Hash)]
pub enum GemvStrategy {
    /// Automatically select best strategy based on dimensions.
    #[default]
    Auto,
    /// Optimized vectorized strategy (fastest for small N / K-contiguous).
    Vectorized,
    /// Scalar strategy (optimized for large N / strided K).
    Scalar,
    /// Canonical 4-way unrolled strategy (legacy compatibility/safety).
    Canonical,
}

/// Unified kernel getter for GEMV V2.
/// Uses centralized KernelRegistry with Arc-based caching.
pub fn get_gemv_v2_kernel(
    policy: std::sync::Arc<dyn crate::fusion::MetalPolicy>,
    layout: Layout,
    strategy: GemvStrategy,
    activation: Activation,
) -> std::sync::Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};

    let variant = format!(
        "{}_{:?}_{}_{}",
        layout.short_name(),
        strategy,
        policy.short_name(),
        activation.struct_name()
    );
    let key = KernelCacheKey::new("gemv", variant);

    let policy_clone = policy.clone();
    kernel_registry().get_or_build(key, move || {
        let kernel_name = format!(
            "gemv_v2_{}_{:?}_{}_{}",
            layout.short_name(),
            strategy,
            policy_clone.short_name(),
            activation.struct_name()
        )
        .to_lowercase();
        let use_f16_cols8 = policy_clone.meta().address_unit_bytes == 2 && use_f16_cols8();

        match (layout, strategy) {
            (Layout::RowMajor, GemvStrategy::Vectorized) | (Layout::RowMajor, GemvStrategy::Auto) => {
                if use_f16_cols8 {
                    CompoundKernel::new(&format!("{}_cols8", kernel_name))
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()).with_f16_cols8(true))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .with_manual_output(true)
                        .compile()
                } else {
                    CompoundKernel::new(&kernel_name)
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .with_manual_output(true)
                        .compile()
                }
            }
            (Layout::RowMajor, GemvStrategy::Canonical) => CompoundKernel::new(&kernel_name)
                .prologue(WarpLayoutStage::row_major().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            (Layout::ColMajor, GemvStrategy::Vectorized) | (Layout::ColMajor, GemvStrategy::Auto) => CompoundKernel::new(&kernel_name)
                .prologue(WarpLayoutStage::col_major().with_warps(8))
                .prologue(VectorizedDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            (Layout::ColMajor, GemvStrategy::Scalar) => CompoundKernel::new(&kernel_name)
                .prologue(ThreadLayoutStage::col_major())
                .prologue(ScalarDotStage::new(policy_clone.clone()))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            (Layout::ColMajor, GemvStrategy::Canonical) => CompoundKernel::new(&kernel_name)
                .prologue(WarpLayoutStage::col_major().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            // Canonical layout: default to the unrolled dot stage (legacy parity + best Q8 decode performance).
            (
                Layout::Canonical {
                    expected_k: _,
                    expected_n: _,
                },
                GemvStrategy::Auto,
            )
            | (
                Layout::Canonical {
                    expected_k: _,
                    expected_n: _,
                },
                GemvStrategy::Canonical,
            ) => CompoundKernel::new(&kernel_name)
                .prologue(WarpLayoutStage::canonical().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            (
                Layout::Canonical {
                    expected_k: _,
                    expected_n: _,
                },
                GemvStrategy::Vectorized,
            ) => CompoundKernel::new(&kernel_name)
                .prologue(WarpLayoutStage::canonical().with_warps(8))
                .prologue(VectorizedDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .with_manual_output(true)
                .compile(),
            _ => panic!("Unsupported layout/strategy pair: {:?}/{:?}", layout, strategy),
        }
    })
}
