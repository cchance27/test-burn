//! GemvV2Step - Full-featured GEMV using Stage composition.
//!
//! Features:
//! - Canonical 4x unrolling (matching legacy performance)
//! - NK/KN layout support via LayoutStage
//! - Policy templates for Q8/F16 transparency
//! - Dynamic block size selection based on K dimension
//! - Composable stage architecture

use std::sync::OnceLock;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel, stages::Layout}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// GemvV2 step - full-featured GEMV using Stage composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvV2Step {
    pub weights: Ref,
    pub scale_bytes: Option<Ref>, // Q8 scales, None for F16
    pub input: Ref,
    pub output: Ref,
    pub bias: Option<Ref>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub weights_per_block: u32, // Typically 32 for Q8
    pub layout: Layout,
    pub strategy: Option<GemvStrategy>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
}

fn default_alpha() -> f32 {
    1.0
}

// =============================================================================
// Fast Warp-Per-Row Kernels (Standard V2)
// =============================================================================

use super::stages::{CanonicalDotStage, VectorizedDotStage, WarpWriteOutputStage};
use crate::compound::stages::{Quantization, WarpLayoutStage, WarpReduceStage};

/// Sub-strategy for GEMV V2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GemvStrategy {
    /// Optimized vectorized strategy (fastest).
    #[default]
    Vectorized,
    /// Canonical 4-way unrolled strategy (legacy compatibility/safety).
    Canonical,
}

/// Get fast warp-per-row kernel for F16.
pub fn get_gemv_v2_kernel_f16(layout: Layout, strategy: GemvStrategy) -> &'static CompiledCompoundKernel {
    match (layout, strategy) {
        (Layout::RowMajor, GemvStrategy::Vectorized) => {
            static NK_KERNEL_F16_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_F16_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_f16_vec")
                    .prologue(WarpLayoutStage::row_major())
                    .prologue(VectorizedDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::RowMajor, GemvStrategy::Canonical) => {
            static NK_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_f16_can")
                    .prologue(WarpLayoutStage::row_major())
                    .prologue(CanonicalDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Vectorized) => {
            static KN_KERNEL_F16_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_F16_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_f16_vec")
                    .prologue(WarpLayoutStage::col_major())
                    .prologue(VectorizedDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Canonical) => {
            static KN_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_f16_can")
                    .prologue(WarpLayoutStage::col_major())
                    .prologue(CanonicalDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::Canonical, GemvStrategy::Vectorized) => {
            static CAN_KERNEL_F16_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            CAN_KERNEL_F16_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_can_f16_vec")
                    .prologue(WarpLayoutStage::canonical())
                    .prologue(VectorizedDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::Canonical, GemvStrategy::Canonical) => {
            static CAN_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            CAN_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_can_f16_can")
                    .prologue(WarpLayoutStage::canonical())
                    .prologue(CanonicalDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
    }
}

/// Get fast warp-per-row kernel for Q8.
pub fn get_gemv_v2_kernel_q8(layout: Layout, strategy: GemvStrategy) -> &'static CompiledCompoundKernel {
    match (layout, strategy) {
        (Layout::RowMajor, GemvStrategy::Vectorized) => {
            static NK_KERNEL_Q8_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_Q8_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_q8_vec")
                    .prologue(WarpLayoutStage::row_major())
                    .prologue(VectorizedDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::RowMajor, GemvStrategy::Canonical) => {
            static NK_KERNEL_Q8_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_Q8_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_q8_can")
                    .prologue(WarpLayoutStage::row_major())
                    .prologue(CanonicalDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Vectorized) => {
            static KN_KERNEL_Q8_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_Q8_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_q8_vec")
                    .prologue(WarpLayoutStage::col_major())
                    .prologue(VectorizedDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Canonical) => {
            static KN_KERNEL_Q8_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_Q8_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_q8_can")
                    .prologue(WarpLayoutStage::col_major())
                    .prologue(CanonicalDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::Canonical, GemvStrategy::Vectorized) => {
            static CAN_KERNEL_Q8_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            CAN_KERNEL_Q8_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_can_q8_vec")
                    .prologue(WarpLayoutStage::canonical())
                    .prologue(VectorizedDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::Canonical, GemvStrategy::Canonical) => {
            static CAN_KERNEL_Q8_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            CAN_KERNEL_Q8_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_can_q8_can")
                    .prologue(WarpLayoutStage::canonical())
                    .prologue(CanonicalDotStage::new(Quantization::Q8))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
    }
}

/// Dispatch configuration for warp-per-row kernels.
/// Returns (grid, group) matching legacy GEMV dispatch.
fn warp_dispatch_config(n_dim: u32) -> DispatchConfig {
    const WARPS_PER_TG: usize = 8;
    const SIMD_WIDTH: usize = 32;
    const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256

    let num_tgs = (n_dim as usize + WARPS_PER_TG - 1) / WARPS_PER_TG;
    DispatchConfig {
        grid: GridSize::new(num_tgs, 1, 1),
        group: ThreadgroupSize::new(TG_WIDTH, 1, 1),
    }
}

#[derive(Debug, Clone)]
pub struct CompiledGemvV2Step {
    pub weights_idx: usize,
    pub scale_bytes_idx: Option<usize>,
    pub input_idx: usize,
    pub output_idx: usize,
    pub bias_idx: Option<usize>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub weights_per_block: u32,
    pub layout: Layout,
    pub is_q8: bool,
    pub strategy: GemvStrategy,
    pub alpha: f32,
}

/// Arguments for GemvV2 kernel dispatch.
/// Buffer layout:
/// 0: weights (uchar* for Policy)
/// 1: scale_bytes (uchar* for Q8, dummy for F16)
/// 2: input (half*)
/// 3: output (half*)
/// 4: K dimension
/// 5: N dimension  
/// 6: weights_per_block
/// 7: bias (half*)
/// 8: has_bias (uint)
#[derive(Debug, KernelArgs)]
struct GemvV2Args {
    #[arg(buffer = 0)]
    weights: TensorArg,
    #[arg(buffer = 1)]
    scale_bytes: TensorArg,
    #[arg(buffer = 2)]
    input: TensorArg,
    #[arg(buffer = 3, output)]
    output: TensorArg,
    #[arg(buffer = 4)]
    k_dim: u32,
    #[arg(buffer = 5)]
    n_dim: u32,
    #[arg(buffer = 6)]
    weights_per_block: u32,
    #[arg(buffer = 7)]
    bias: TensorArg,
    #[arg(buffer = 8)]
    has_bias: u32,
    #[arg(buffer = 9)]
    alpha: f32,
}

#[typetag::serde(name = "GemvV2")]
impl Step for GemvV2Step {
    fn name(&self) -> &'static str {
        "GemvV2"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("GemvV2 only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let weights_idx = symbols.get_or_create(bindings.interpolate(self.weights.0.clone()));
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        let scale_bytes_idx = self
            .scale_bytes
            .as_ref()
            .map(|s| symbols.get_or_create(bindings.interpolate(s.0.clone())));
        let bias_idx = self.bias.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));

        // Determine quantization mode based on scale_bytes presence
        let is_q8 = self.scale_bytes.is_some();

        // Strategy priority:
        // 1. Explicitly set in struct
        // 2. Environment variable override
        // 3. Default (Vectorized)
        let strategy = self.strategy.unwrap_or_else(|| {
            if std::env::var("METALLIC_GEMV_STRATEGY").ok().as_deref() == Some("canonical") {
                GemvStrategy::Canonical
            } else {
                GemvStrategy::Vectorized
            }
        });

        // Ensure kernel is compiled (pre-warm cache)
        if is_q8 {
            get_gemv_v2_kernel_q8(self.layout, strategy);
        } else {
            get_gemv_v2_kernel_f16(self.layout, strategy);
        }

        vec![Box::new(CompiledGemvV2Step {
            weights_idx,
            scale_bytes_idx,
            input_idx,
            output_idx,
            bias_idx,
            k_dim: self.k_dim.clone(),
            n_dim: self.n_dim.clone(),
            weights_per_block: self.weights_per_block,
            layout: self.layout,
            is_q8,
            strategy,
            alpha: self.alpha,
        })]
    }
}

impl CompiledStep for CompiledGemvV2Step {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, bindings: &TensorBindings) -> Result<(), MetalError> {
        let weights = fast_bindings
            .get(self.weights_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Weights {}", self.weights_idx)))?;
        let input = fast_bindings
            .get(self.input_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Input {}", self.input_idx)))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Output {}", self.output_idx)))?;

        // Resolve dimensions
        let k_dim = self.k_dim.resolve(bindings);
        let n_dim = self.n_dim.resolve(bindings);

        // Scale bytes: use provided or duplicate weights (F16 case - Policy ignores)
        let scale_bytes = if let Some(idx) = self.scale_bytes_idx {
            let s = fast_bindings
                .get(idx)
                .ok_or_else(|| MetalError::InputNotFound(format!("ScaleBytes {}", idx)))?;
            TensorArg::from_tensor(s)
        } else {
            // F16 mode: PolicyF16::load_scale returns 1.0, so this is unused
            TensorArg::from_tensor(weights)
        };

        // Handle optional bias
        let (bias, has_bias) = if let Some(idx) = self.bias_idx {
            let b = fast_bindings
                .get(idx)
                .ok_or_else(|| MetalError::InputNotFound(format!("Bias {}", idx)))?;
            (TensorArg::from_tensor(b), 1u32)
        } else {
            // Use output as placeholder for missing bias
            (TensorArg::from_tensor(output), 0u32)
        };

        let args = GemvV2Args {
            weights: TensorArg::from_tensor(weights),
            scale_bytes,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block: self.weights_per_block,
            bias,
            has_bias,
            alpha: self.alpha,
        };

        // Choose dispatch config (always warp-per-row now)
        let dispatch = warp_dispatch_config(n_dim);

        // Select kernel
        let kernel = if self.is_q8 {
            get_gemv_v2_kernel_q8(self.layout, self.strategy)
        } else {
            get_gemv_v2_kernel_f16(self.layout, self.strategy)
        };

        let bound_kernel = kernel.bind(args, dispatch);
        foundry.run(&bound_kernel)
    }
}
