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

use super::{
    config::use_f16_cols8, stages::{CanonicalDotStage, ScalarDotStage, VectorizedDotStage, WarpWriteOutputStage}
};
use crate::{
    MetalError, compound::{
        CompiledCompoundKernel, Layout, stages::{ThreadLayoutStage, WarpLayoutStage, WarpReduceStage}
    }, metals::common::{cache::get_or_build_compound_kernel, composition::manual_output}, policy::activation::Activation, types::TensorArg
};

// =============================================================================
// Parameters and Arguments
// =============================================================================

/// Arguments for GemvV2 kernel dispatch.
#[derive(Debug, KernelArgs)]
pub struct GemvV2Args {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 2, metal_type = "const device InputStorageT*")]
    pub input: TensorArg,
    #[arg(buffer = 3, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 4)]
    pub k_dim: u32,
    #[arg(buffer = 5)]
    pub n_dim: u32,
    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    #[arg(buffer = 7, metal_type = "const device BiasStorageT*")]
    pub bias: TensorArg,
    #[arg(buffer = 8)]
    pub has_bias: u32,
    #[arg(buffer = 9)]
    pub alpha: f32,
    #[arg(buffer = 10, metal_type = "const device ResidualStorageT*")]
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
    /// Decode LM-head optimized vectorized strategy (very large N, m=1 row-major path).
    DecodeLmHead,
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
) -> Result<std::sync::Arc<CompiledCompoundKernel>, MetalError> {
    let supported_pair = matches!(
        (layout, strategy),
        (Layout::RowMajor, GemvStrategy::Vectorized)
            | (Layout::RowMajor, GemvStrategy::Auto)
            | (Layout::RowMajor, GemvStrategy::DecodeLmHead)
            | (Layout::RowMajor, GemvStrategy::Canonical)
            | (Layout::ColMajor, GemvStrategy::Vectorized)
            | (Layout::ColMajor, GemvStrategy::Auto)
            | (Layout::ColMajor, GemvStrategy::Scalar)
            | (Layout::ColMajor, GemvStrategy::Canonical)
            | (Layout::Canonical { .. }, GemvStrategy::Auto)
            | (Layout::Canonical { .. }, GemvStrategy::Canonical)
            | (Layout::Canonical { .. }, GemvStrategy::Vectorized)
    );
    if !supported_pair {
        return Err(MetalError::OperationNotSupported(format!(
            "Unsupported GEMV layout/strategy pair: layout={layout:?}, strategy={strategy:?}"
        )));
    }

    let variant = format!(
        "{}_{:?}_{}_{}",
        layout.short_name(),
        strategy,
        policy.short_name(),
        activation.struct_name()
    );
    let policy_clone = policy.clone();
    Ok(get_or_build_compound_kernel("gemv", variant, move || {
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
            (Layout::RowMajor, GemvStrategy::DecodeLmHead) => {
                if use_f16_cols8 {
                    manual_output(&format!("{}_cols8", kernel_name))
                        .prologue(WarpLayoutStage::row_major().with_warps(16))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()).with_f16_cols8(true))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .compile()
                } else {
                    manual_output(&kernel_name)
                        .prologue(WarpLayoutStage::row_major().with_warps(16))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .compile()
                }
            }
            (Layout::RowMajor, GemvStrategy::Vectorized) | (Layout::RowMajor, GemvStrategy::Auto) => {
                if use_f16_cols8 {
                    manual_output(&format!("{}_cols8", kernel_name))
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()).with_f16_cols8(true))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .compile()
                } else {
                    manual_output(&kernel_name)
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(policy_clone.clone()))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new().with_activation(activation))
                        .compile()
                }
            }
            (Layout::RowMajor, GemvStrategy::Canonical) => manual_output(&kernel_name)
                .prologue(WarpLayoutStage::row_major().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .compile(),
            (Layout::ColMajor, GemvStrategy::Vectorized) | (Layout::ColMajor, GemvStrategy::Auto) => manual_output(&kernel_name)
                .prologue(WarpLayoutStage::col_major().with_warps(8))
                .prologue(VectorizedDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .compile(),
            (Layout::ColMajor, GemvStrategy::Scalar) => manual_output(&kernel_name)
                .prologue(ThreadLayoutStage::col_major())
                .prologue(ScalarDotStage::new(policy_clone.clone()))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .compile(),
            (Layout::ColMajor, GemvStrategy::Canonical) => manual_output(&kernel_name)
                .prologue(WarpLayoutStage::col_major().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
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
            ) => manual_output(&kernel_name)
                .prologue(WarpLayoutStage::canonical().with_warps(8))
                .prologue(CanonicalDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .compile(),
            (
                Layout::Canonical {
                    expected_k: _,
                    expected_n: _,
                },
                GemvStrategy::Vectorized,
            ) => manual_output(&kernel_name)
                .prologue(WarpLayoutStage::canonical().with_warps(8))
                .prologue(VectorizedDotStage::new(policy_clone.clone()))
                .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                .main(WarpWriteOutputStage::new().with_activation(activation))
                .compile(),
            _ => unreachable!("unsupported layout/strategy pair validated before kernel build"),
        }
    }))
}
