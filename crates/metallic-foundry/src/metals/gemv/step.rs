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

use super::stages::{CanonicalDotStage, ScalarDotStage, VectorizedDotStage, WarpWriteOutputStage};
use crate::{
    Foundry, MetalError, compound::{
        BufferArg, CompiledCompoundKernel, CompoundKernel, Stage, stages::{Layout, Quantization, ThreadLayoutStage, WarpLayoutStage, WarpReduceStage}
    }, spec::{CompiledStep, DynamicValue, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

fn use_f16_cols8() -> bool {
    // Default ON: this path mirrors the legacy Context RowMajor FP16 GEMV pointer arithmetic and is consistently faster
    // for decode-heavy shapes (e.g. K=896, K=4864). Allow an escape hatch to disable for debugging/regressions.
    std::env::var("METALLIC_GEMV_F16_COLS8").ok().map(|val| val != "0").unwrap_or(true)
}

/// GemvV2 step - full-featured GEMV using Stage composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvV2Step {
    pub weights: Ref,
    pub scale_bytes: Option<Ref>, // Q8 scales, None for F16
    pub input: Ref,
    pub output: Ref,
    pub bias: Option<Ref>,
    pub residual: Option<Ref>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub weights_per_block: u32, // Typically 32 for Q8
    pub layout: Layout,
    pub strategy: Option<GemvStrategy>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default)]
    pub beta: f32,
}

fn default_alpha() -> f32 {
    1.0
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

/// Get fast warp-per-row kernel for F16.
pub fn get_gemv_v2_kernel_f16(layout: Layout, strategy: GemvStrategy) -> &'static CompiledCompoundKernel {
    match (layout, strategy) {
        (Layout::RowMajor, GemvStrategy::Vectorized) | (Layout::RowMajor, GemvStrategy::Auto) => {
            if use_f16_cols8() {
                static NK_KERNEL_F16_COLS8: OnceLock<CompiledCompoundKernel> = OnceLock::new();
                NK_KERNEL_F16_COLS8.get_or_init(|| {
                    CompoundKernel::new("gemv_v2_nk_f16_cols8")
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(Quantization::F16).with_f16_cols8(true))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new())
                        .with_manual_output(true)
                        .compile()
                })
            } else {
                static NK_KERNEL_F16_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
                NK_KERNEL_F16_VEC.get_or_init(|| {
                    CompoundKernel::new("gemv_v2_nk_f16_vec")
                        .prologue(WarpLayoutStage::row_major().with_warps(8))
                        .prologue(VectorizedDotStage::new(Quantization::F16))
                        .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                        .main(WarpWriteOutputStage::new())
                        .with_manual_output(true)
                        .compile()
                })
            }
        }
        (Layout::RowMajor, GemvStrategy::Canonical) => {
            static NK_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_f16_can")
                    .prologue(WarpLayoutStage::row_major().with_warps(8))
                    .prologue(CanonicalDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Vectorized) | (Layout::ColMajor, GemvStrategy::Auto) => {
            // Auto defaults to Vectorized for ColMajor unless specifically hitting Large N path at runtime check
            static KN_KERNEL_F16_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_F16_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_f16_vec")
                    .prologue(WarpLayoutStage::col_major().with_warps(8))
                    .prologue(VectorizedDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Scalar) => {
            static KN_KERNEL_F16_SCALAR: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_F16_SCALAR.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_f16_scalar")
                    .prologue(ThreadLayoutStage::col_major())
                    .prologue(ScalarDotStage::new(Quantization::F16))
                    // No WarpReduce needed, ScalarDot computes full sum
                    .main(WarpWriteOutputStage::new()) // Works because lane_id=0 defined by ThreadLayout
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::ColMajor, GemvStrategy::Canonical) => {
            static KN_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_f16_can")
                    .prologue(WarpLayoutStage::col_major().with_warps(8))
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
                    .prologue(WarpLayoutStage::canonical().with_warps(8))
                    .prologue(VectorizedDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        (Layout::Canonical, GemvStrategy::Canonical) | (Layout::Canonical, GemvStrategy::Auto) => {
            static CAN_KERNEL_F16_CAN: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            CAN_KERNEL_F16_CAN.get_or_init(|| {
                CompoundKernel::new("gemv_v2_can_f16_can")
                    .prologue(WarpLayoutStage::canonical().with_warps(8))
                    .prologue(CanonicalDotStage::new(Quantization::F16))
                    .prologue(WarpReduceStage::sum("partial_dot", "row_sum"))
                    .main(WarpWriteOutputStage::new())
                    .with_manual_output(true)
                    .compile()
            })
        }
        _ => unreachable!("Unsupported layout/strategy pair for F16"),
    }
}

/// Get fast warp-per-row kernel for Q8.
pub fn get_gemv_v2_kernel_q8(layout: Layout, strategy: GemvStrategy) -> &'static CompiledCompoundKernel {
    match (layout, strategy) {
        (Layout::RowMajor, GemvStrategy::Vectorized) | (Layout::RowMajor, GemvStrategy::Auto) => {
            static NK_KERNEL_Q8_VEC: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            NK_KERNEL_Q8_VEC.get_or_init(|| {
                CompoundKernel::new("gemv_v2_nk_q8_vec")
                    .prologue(WarpLayoutStage::row_major().with_warps(8))
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
        (Layout::ColMajor, GemvStrategy::Vectorized) | (Layout::ColMajor, GemvStrategy::Auto) => {
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
        (Layout::ColMajor, GemvStrategy::Scalar) => {
            static KN_KERNEL_Q8_SCALAR: OnceLock<CompiledCompoundKernel> = OnceLock::new();
            KN_KERNEL_Q8_SCALAR.get_or_init(|| {
                CompoundKernel::new("gemv_v2_kn_q8_scalar")
                    .prologue(ThreadLayoutStage::col_major())
                    .prologue(ScalarDotStage::new(Quantization::Q8))
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
        (Layout::Canonical, GemvStrategy::Canonical) | (Layout::Canonical, GemvStrategy::Auto) => {
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
        _ => unreachable!("Unsupported layout/strategy pair for Q8"),
    }
}

/// Dispatch configuration for warp-per-row kernels.
/// Returns (grid, group) matching legacy GEMV dispatch.
pub fn warp_dispatch_config(n_dim: u32) -> DispatchConfig {
    const WARPS_PER_TG: usize = 8;
    const SIMD_WIDTH: usize = 32;
    const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256

    let num_tgs = (n_dim as usize).div_ceil(WARPS_PER_TG);
    DispatchConfig {
        grid: GridSize::new(num_tgs, 1, 1),
        group: ThreadgroupSize::new(TG_WIDTH, 1, 1),
    }
}

/// Dispatch configuration for warp-per-row kernels with a batch (M) dimension.
///
/// Uses `gid.y` as the batch index.
pub fn warp_dispatch_config_2d(n_dim: u32, batch: u32) -> DispatchConfig {
    const WARPS_PER_TG: usize = 8;
    const SIMD_WIDTH: usize = 32;
    const TG_WIDTH: usize = WARPS_PER_TG * SIMD_WIDTH; // 256

    let num_tgs = (n_dim as usize).div_ceil(WARPS_PER_TG);
    DispatchConfig {
        grid: GridSize::new(num_tgs, (batch.max(1)) as usize, 1),
        group: ThreadgroupSize::new(TG_WIDTH, 1, 1),
    }
}

/// Dispatch configuration for thread-per-row (scalar) kernels.
pub fn thread_dispatch_config(output_rows: u32) -> DispatchConfig {
    let threads_per_tg = 256;
    let threadgroups = output_rows.div_ceil(threads_per_tg);

    DispatchConfig {
        grid: GridSize::d1(threadgroups as usize),
        group: ThreadgroupSize::d1(threads_per_tg as usize),
    }
}

#[derive(Debug, Clone)]
pub struct CompiledGemvV2Step {
    pub weights_name: String,
    pub weights_resolved: ResolvedSymbols,
    pub scale_bytes_idx: Option<usize>,
    pub input_idx: usize,
    pub output_idx: usize,
    pub bias_idx: Option<usize>,
    pub residual_idx: Option<usize>,
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub weights_per_block: u32,
    pub layout: Layout,
    pub strategy: GemvStrategy,
    pub alpha: f32,
    pub beta: f32,
}

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

#[typetag::serde(name = "GemvV2")]
impl Step for GemvV2Step {
    fn name(&self) -> &'static str {
        "GemvV2"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("GemvV2 only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let weights_name = bindings.interpolate(self.weights.0.clone());
        let input_name = bindings.interpolate(self.input.0.clone());
        let output_name = bindings.interpolate(self.output.0.clone());

        let weights_idx = symbols.get_or_create(weights_name.clone());
        let input_idx = symbols.get_or_create(input_name);
        let output_idx = symbols.get_or_create(output_name);

        let scale_bytes_idx = self
            .scale_bytes
            .as_ref()
            .map(|s| symbols.get_or_create(bindings.interpolate(s.0.clone())));
        let derived_scale_bytes_idx = symbols.get_or_create(format!("{weights_name}_scales"));
        let bias_idx = self.bias.as_ref().map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let residual_idx = self
            .residual
            .as_ref()
            .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));

        // Strategy priority:
        // 1. Explicitly set in struct
        // 2. Environment variable override
        // 3. Default (Vectorized)
        let strategy = self.strategy.unwrap_or_else(|| {
            if std::env::var("METALLIC_GEMV_STRATEGY").ok().as_deref() == Some("canonical") {
                GemvStrategy::Canonical
            } else if std::env::var("METALLIC_GEMV_STRATEGY").ok().as_deref() == Some("scalar") {
                GemvStrategy::Scalar
            } else {
                GemvStrategy::Auto
            }
        });

        // Pre-warm both variants; quantization is selected at runtime from the bound weight dtype.
        get_gemv_v2_kernel_f16(self.layout, strategy);
        get_gemv_v2_kernel_q8(self.layout, strategy);

        vec![Box::new(CompiledGemvV2Step {
            weights_name,
            weights_resolved: ResolvedSymbols {
                weights: weights_idx,
                scales: derived_scale_bytes_idx.into(), // Using pre-resolved scales index
                bias: None,
            },
            scale_bytes_idx,
            input_idx,
            output_idx,
            bias_idx,
            residual_idx,
            k_dim: self.k_dim.clone(),
            n_dim: self.n_dim.clone(),
            weights_per_block: self.weights_per_block,
            layout: self.layout,
            strategy,
            alpha: self.alpha,
            beta: self.beta,
        })]
    }
}

impl CompiledStep for CompiledGemvV2Step {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let weights = fast_bindings
            .get(self.weights_resolved.weights)
            .ok_or_else(|| MetalError::InputNotFound(format!("Weights {}", self.weights_resolved.weights)))?;
        let input = fast_bindings
            .get(self.input_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Input {}", self.input_idx)))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound(format!("Output {}", self.output_idx)))?;

        // Resolve dimensions
        let k_dim = self.k_dim.resolve(bindings);
        let n_dim = self.n_dim.resolve(bindings);

        // Centralized Quantization Binding
        let policy = crate::policy::resolve_policy(weights.dtype.into());
        let loader = policy.loader_stage();
        let quantization = loader.quantization_type();
        let loader_args = loader.bind(fast_bindings, &self.weights_resolved);

        let weights_arg = loader_args[0].clone();
        let scale_bytes = loader_args[1].clone();
        let is_q8 = quantization == Quantization::Q8;

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

        // Handle optional residual
        let (residual, has_residual) = if let Some(idx) = self.residual_idx {
            let r = fast_bindings
                .get(idx)
                .ok_or_else(|| MetalError::InputNotFound(format!("Residual {}", idx)))?;
            (TensorArg::from_tensor(r), 1u32)
        } else {
            (TensorArg::from_tensor(output), 0u32)
        };

        // let weights_arg = TensorArg::from_tensor(weights); // Handled by loader_args[0] above

        let batch = bindings.get_int_global("m").unwrap_or(1).max(1) as u32;

        // Determine Effective Strategy
        let mut effective_strategy = match self.strategy {
            GemvStrategy::Auto => {
                // If Auto, choose Scalar for Large N in ColMajor layout
                // Using cutoff N > 4096 based on typical Warp vs Thread efficiency
                if self.layout == Layout::ColMajor && n_dim > 4096 {
                    GemvStrategy::Scalar
                } else {
                    GemvStrategy::Vectorized
                }
            }
            s => s,
        };

        // Batched dispatch requires WarpLayoutStage semantics (uses gid.y = batch_idx).
        if batch > 1 && effective_strategy == GemvStrategy::Scalar {
            effective_strategy = GemvStrategy::Vectorized;
        }

        // Select Kernel based on effective strategy
        let kernel = if is_q8 {
            get_gemv_v2_kernel_q8(self.layout, effective_strategy)
        } else {
            get_gemv_v2_kernel_f16(self.layout, effective_strategy)
        };

        let args = GemvV2Args {
            weights: weights_arg,
            scale_bytes,
            input: TensorArg::from_tensor(input),
            output: TensorArg::from_tensor(output),
            k_dim,
            n_dim,
            weights_per_block: self.weights_per_block,
            bias,
            has_bias,
            alpha: self.alpha,
            residual,
            has_residual,
            beta: self.beta,
        };

        // Dispatch Logic based on Strategy
        let dispatch = if effective_strategy == GemvStrategy::Scalar {
            thread_dispatch_config(n_dim)
        } else {
            warp_dispatch_config_2d(n_dim, batch)
        };

        if crate::instrument::emit_cb_timing_metrics() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), batch.to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n_dim.to_string());
            data.insert("k".to_string(), k_dim.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "1".to_string());
            foundry.run(&kernel.bind_with_metrics(args, dispatch, data))?;
        } else {
            foundry.run(&kernel.bind(args, dispatch))?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "GemvV2"
    }
}

// =============================================================================
// Legacy GemvCanonical and GemvColMajor Step Wrappers
// These map the old interface to GemvV2
// =============================================================================

/// Legacy GemvCanonical Step for DSL compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvCanonicalStep {
    pub matrix: Ref,
    pub scale_bytes: Option<Ref>,
    pub vector_x: Ref,
    pub result_y: Ref,
    pub bias: Option<Ref>,
    pub residual: Option<Ref>,
    #[serde(default)]
    pub params: GemvLegacyParams,
    #[serde(default = "default_alpha_legacy")]
    pub alpha: f32,
    #[serde(default)]
    pub beta: f32,
    #[serde(default)]
    pub has_bias: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GemvLegacyParams {
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default = "default_wpb_legacy")]
    pub weights_per_block: u32,
}

fn default_alpha_legacy() -> f32 {
    1.0
}
fn default_batch() -> u32 {
    1
}
fn default_wpb_legacy() -> u32 {
    32
}

#[derive(Debug, Clone)]
pub struct CompiledGemvCanonicalStep {
    pub step: GemvCanonicalStep,
    pub matrix_name: String,
    pub matrix_resolved: ResolvedSymbols,
    pub vector_x_idx: usize,
    pub result_y_idx: usize,
    pub bias_idx: Option<usize>,
    pub residual_idx: Option<usize>,
}

#[typetag::serde(name = "GemvCanonical")]
impl Step for GemvCanonicalStep {
    fn name(&self) -> &'static str {
        "GemvCanonical"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        // Bind all symbols found in the table
        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let matrix_name = bindings.interpolate(self.matrix.0.clone());
        let matrix_idx = symbols.get_or_create(matrix_name.clone());
        let _matrix_scales_idx = symbols.get_or_create(format!("{matrix_name}_scales"));
        let vector_x_idx = symbols.get_or_create(bindings.interpolate(self.vector_x.0.clone()));
        let result_y_idx = symbols.get_or_create(bindings.interpolate(self.result_y.0.clone()));
        let bias_idx = self
            .bias
            .as_ref()
            .filter(|b| b.0 != "zero")
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let residual_idx = self
            .residual
            .as_ref()
            .filter(|r| r.0 != "zero")
            .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));

        vec![Box::new(CompiledGemvCanonicalStep {
            step: self.clone(),
            matrix_name,
            matrix_resolved: ResolvedSymbols {
                weights: matrix_idx,
                scales: _matrix_scales_idx.into(),
                bias: None,
            },
            vector_x_idx,
            result_y_idx,
            bias_idx,
            residual_idx,
        })]
    }
}

impl CompiledStep for CompiledGemvCanonicalStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let matrix = fast_bindings
            .get(self.matrix_resolved.weights)
            .ok_or(MetalError::InputNotFound("matrix".into()))?;
        let vector_x = fast_bindings
            .get(self.vector_x_idx)
            .ok_or(MetalError::InputNotFound("vector_x".into()))?;
        let result_y = fast_bindings
            .get(self.result_y_idx)
            .ok_or(MetalError::InputNotFound("result_y".into()))?;

        // Centralized Quantization Binding
        let policy = crate::policy::resolve_policy(matrix.dtype.into());
        let loader = policy.loader_stage();
        let quantization = loader.quantization_type();

        let matrix_args = loader.bind(fast_bindings, &self.matrix_resolved);

        let weights = matrix_args[0].clone();
        let scale_arg = matrix_args[1].clone();

        // Handle Bias
        let (bias, has_bias) = if let Some(idx) = self.bias_idx {
            let b = fast_bindings.get(idx).ok_or(MetalError::InputNotFound("bias".into()))?;
            (TensorArg::from_tensor(b), 1)
        } else {
            (TensorArg::from_tensor(result_y), 0)
        };

        // Handle optional residual (fused in write stage)
        let (residual, has_residual) = if let Some(res_idx) = self.residual_idx {
            let r = fast_bindings.get(res_idx).ok_or(MetalError::InputNotFound("residual".into()))?;
            (TensorArg::from_tensor(r), 1)
        } else {
            (TensorArg::from_tensor(result_y), 0)
        };

        // Dimensions detection: Check against vector_x (input) length K
        let x_k = vector_x.dims.last().copied().unwrap_or(0) as u32;
        let (n_dim, k_dim) = if matrix.dims.len() >= 2 {
            let d0 = matrix.dims[0] as u32;
            let d1 = matrix.dims[1] as u32;

            if d0 == x_k && d1 != x_k {
                // [K, N] layout (CanonicalF16Tensor standard)
                (d1, d0)
            } else if d1 == x_k && d0 != x_k {
                // [N, K] layout (Standard RowMajor)
                (d0, d1)
            } else {
                // Square or Ambiguous - fall back to assuming [N, K] or [K, N] order dependent on struct
                // For square [K, K], (d0, d1) works.
                (d0, d1)
            }
        } else {
            let total = matrix.dims.iter().product::<usize>() as u32;
            if x_k == 0 {
                return Err(MetalError::InvalidShape(
                    "Cannot infer GemvCanonical dimensions: vector_x has 0 dims".into(),
                ));
            }
            let n = total / x_k;
            (n, x_k)
        };

        // let weights = TensorArg::from_tensor(matrix); // Handled by matrix_args[0]
        /*
        let scale_arg = if self.is_q8 {
            TensorArg::from_tensor(scale_bytes)
        } else {
            TensorArg::from_tensor(matrix) // Dummy bind to valid buffer
        };
        */

        let args = GemvV2Args {
            weights,
            scale_bytes: scale_arg,
            input: TensorArg::from_tensor(vector_x),
            output: TensorArg::from_tensor(result_y),
            k_dim,
            n_dim,
            weights_per_block: self.step.params.weights_per_block,
            bias,
            has_bias,
            alpha: self.step.alpha,
            residual,
            has_residual,
            beta: self.step.beta,
        };

        let batch = bindings.get_int_global("m").unwrap_or(self.step.params.batch as usize).max(1) as u32;
        let dispatch = warp_dispatch_config_2d(n_dim, batch);

        let kernel = match quantization {
            Quantization::Q8 => get_gemv_v2_kernel_q8(crate::compound::stages::Layout::Canonical, GemvStrategy::Canonical),
            _ => get_gemv_v2_kernel_f16(crate::compound::stages::Layout::Canonical, GemvStrategy::Canonical),
        };

        if crate::instrument::emit_cb_timing_metrics() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), batch.to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n_dim.to_string());
            data.insert("k".to_string(), k_dim.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "1".to_string());
            foundry.run(&kernel.bind_with_metrics(args, dispatch, data))?;
        } else {
            foundry.run(&kernel.bind(args, dispatch))?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "GemvCanonical"
    }
}

/// Legacy GemvColMajor Step for DSL compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvColMajorStep {
    pub matrix: Ref,
    pub scale_bytes: Option<Ref>,
    pub vector_x: Ref,
    pub result_y: Ref,
    pub bias: Option<Ref>,
    pub residual: Option<Ref>,
    #[serde(default)]
    pub params: GemvColMajorParams,
    #[serde(default = "default_alpha_legacy")]
    pub alpha: f32,
    #[serde(default)]
    pub beta: f32,
    #[serde(default)]
    pub has_bias: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GemvColMajorParams {
    #[serde(default)]
    pub n: u32,
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default)]
    pub fused_bias_offset: u32,
}

#[derive(Debug, Clone)]
pub struct CompiledGemvColMajorStep {
    pub step: GemvColMajorStep,
    pub matrix_name: String,
    pub matrix_resolved: ResolvedSymbols,
    pub vector_x_idx: usize,
    pub result_y_idx: usize,
    pub bias_idx: Option<usize>,
    pub residual_idx: Option<usize>,
}

#[typetag::serde(name = "GemvColMajor")]
impl Step for GemvColMajorStep {
    fn name(&self) -> &'static str {
        "GemvColMajor"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        // Bind all symbols found in the table
        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let matrix_name = bindings.interpolate(self.matrix.0.clone());
        let matrix_idx = symbols.get_or_create(matrix_name.clone());
        let _matrix_scales_idx = symbols.get_or_create(format!("{matrix_name}_scales"));
        let vector_x_idx = symbols.get_or_create(bindings.interpolate(self.vector_x.0.clone()));
        let result_y_idx = symbols.get_or_create(bindings.interpolate(self.result_y.0.clone()));
        let bias_idx = self
            .bias
            .as_ref()
            .filter(|b| b.0 != "zero")
            .map(|b| symbols.get_or_create(bindings.interpolate(b.0.clone())));
        let residual_idx = self
            .residual
            .as_ref()
            .filter(|r| r.0 != "zero")
            .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));

        vec![Box::new(CompiledGemvColMajorStep {
            step: self.clone(),
            matrix_name,
            matrix_resolved: ResolvedSymbols {
                weights: matrix_idx,
                scales: _matrix_scales_idx.into(),
                bias: None,
            },
            vector_x_idx,
            result_y_idx,
            bias_idx,
            residual_idx,
        })]
    }
}

impl CompiledStep for CompiledGemvColMajorStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        // Debug flag for granular tracing
        let debug = bindings.get_var("DEBUG").is_some();

        if debug {
            eprintln!("  [GemvColMajor] Resolving tensors...");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        let matrix = fast_bindings
            .get(self.matrix_resolved.weights)
            .ok_or(MetalError::InputNotFound("matrix".into()))?;
        let vector_x = fast_bindings
            .get(self.vector_x_idx)
            .ok_or(MetalError::InputNotFound("vector_x".into()))?;
        let result_y = fast_bindings
            .get(self.result_y_idx)
            .ok_or(MetalError::InputNotFound("result_y".into()))?;

        // Centralized Quantization Binding
        let policy = crate::policy::resolve_policy(matrix.dtype.into());
        let loader = policy.loader_stage();
        let quantization = loader.quantization_type();

        let matrix_args = loader.bind(fast_bindings, &self.matrix_resolved);

        let weights = matrix_args[0].clone();
        let scale_arg = matrix_args[1].clone();
        let is_q8 = quantization == Quantization::Q8;

        if debug {
            eprintln!(
                "  [GemvColMajor] Tensors resolved. matrix_dims={:?}, vector_x_dims={:?}, result_y_dims={:?}",
                matrix.dims, vector_x.dims, result_y.dims
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        // Handle Bias
        let (bias, has_bias) = if let Some(idx) = self.bias_idx {
            let b = fast_bindings.get(idx).ok_or(MetalError::InputNotFound("bias".into()))?;
            (TensorArg::from_tensor(b), 1)
        } else {
            (TensorArg::from_tensor(result_y), 0)
        };

        // Dimensions: ColMajor stores matrix as [N, K] where:
        // - N = output dimension (rows = result_y size)
        // - K = input dimension (cols = vector_x size)
        // matrix_dims[0] = N (output), matrix_dims[1] = K (input)
        let (n_dim, k_dim) = if matrix.dims.len() >= 2 {
            (matrix.dims[0] as u32, matrix.dims[1] as u32)
        } else {
            let total = matrix.dims.iter().product::<usize>() as u32;
            let k = vector_x.dims.last().copied().unwrap_or(0) as u32;
            if k == 0 {
                return Err(MetalError::InvalidShape(
                    "Cannot infer GemvColMajor dimensions: vector_x has 0 dims".into(),
                ));
            }
            let n = total / k;
            (n, k)
        };

        if debug {
            eprintln!("  [GemvColMajor] Dimensions: k_dim={}, n_dim={}, is_q8={}", k_dim, n_dim, is_q8);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        // let weights = TensorArg::from_tensor(matrix); // Handled by matrix_args[0]
        /*
        let scale_arg = if self.is_q8 {
            TensorArg::from_tensor(scale_bytes)
        } else {
            TensorArg::from_tensor(matrix)
        };
        */

        let args = GemvV2Args {
            weights,
            scale_bytes: scale_arg,
            input: TensorArg::from_tensor(vector_x),
            output: TensorArg::from_tensor(result_y),
            k_dim,
            n_dim,
            weights_per_block: 32,
            bias,
            has_bias,
            alpha: self.step.alpha,
            residual: TensorArg::from_tensor(result_y),
            has_residual: 0,
            beta: 0.0,
        };

        let dispatch = if n_dim > 4096 {
            // Use scalar dispatch (thread-per-row) for large N (ColMajor)
            thread_dispatch_config(n_dim)
        } else {
            // Use typical warp dispatch for small N
            warp_dispatch_config(n_dim)
        };

        if debug {
            eprintln!("  [GemvColMajor] Getting kernel...");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        // Use Vectorized strategy for ColMajor as it naturally vectorizes
        // Logic Update: Use Scalar if N > 4096
        let kernel = match quantization {
            Quantization::Q8 => {
                if n_dim > 4096 {
                    get_gemv_v2_kernel_q8(crate::compound::stages::Layout::ColMajor, GemvStrategy::Scalar)
                } else {
                    get_gemv_v2_kernel_q8(crate::compound::stages::Layout::ColMajor, GemvStrategy::Vectorized)
                }
            }
            _ => {
                if n_dim > 4096 {
                    get_gemv_v2_kernel_f16(crate::compound::stages::Layout::ColMajor, GemvStrategy::Scalar)
                } else {
                    get_gemv_v2_kernel_f16(crate::compound::stages::Layout::ColMajor, GemvStrategy::Vectorized)
                }
            }
        };

        if debug {
            eprintln!("  [GemvColMajor] Running kernel dispatch...");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        if crate::instrument::emit_cb_timing_metrics() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n_dim.to_string());
            data.insert("k".to_string(), k_dim.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "1".to_string());
            foundry.run(&kernel.bind_with_metrics(args, dispatch, data))?;
        } else {
            foundry.run(&kernel.bind(args, dispatch))?;
        }

        if debug {
            eprintln!("  [GemvColMajor] Kernel dispatched successfully.");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        if let Some(res_idx) = self.residual_idx {
            let residual = fast_bindings.get(res_idx).ok_or(MetalError::InputNotFound("residual".into()))?;

            let total = result_y.dims.iter().product::<usize>() as u32;
            let add_args = AddArgs {
                a: TensorArg::from_tensor(result_y),
                b: TensorArg::from_tensor(residual),
                out: TensorArg::from_tensor(result_y),
                total,
            };

            let add_kernel = get_add_kernel();
            let grid_size = total.div_ceil(256) as usize;
            let dispatch = DispatchConfig {
                grid: GridSize::d1(grid_size),
                group: ThreadgroupSize::d1(256),
            };

            foundry.run(&add_kernel.bind(add_args, dispatch))?;
        }

        if debug {
            eprintln!("  [GemvColMajor] Execute complete.");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "GemvColMajor"
    }
}

// =============================================================================
// Helper: Add Kernel for Residual Accumulation
// =============================================================================

#[derive(Debug, KernelArgs)]
struct AddArgs {
    #[arg(buffer = 0)]
    a: TensorArg,
    #[arg(buffer = 1)]
    b: TensorArg,
    #[arg(buffer = 2)]
    out: TensorArg,
    #[arg(buffer = 3)]
    total: u32,
}

#[derive(Debug, Clone, KernelArgs)]
struct ElemwiseAddGlobalStage;

impl Stage for ElemwiseAddGlobalStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }
    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "a",
                metal_type: "const device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "b",
                metal_type: "const device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "out",
                metal_type: "device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "total_elements",
                metal_type: "constant uint&",
                buffer_index: 3,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        String::new()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        (
            "void".to_string(),
            r#"
    idx = gid.x * tptg.x + lid.x;
    if (idx >= total_elements) return;
    out[idx] = a[idx] + b[idx];
            "#
            .to_string(),
        )
    }
}

fn get_add_kernel() -> &'static CompiledCompoundKernel {
    static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        CompoundKernel::new("gemv_add_residual")
            .main(ElemwiseAddGlobalStage)
            .with_manual_output(true)
            .compile()
    })
}
