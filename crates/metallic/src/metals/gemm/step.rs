//! GEMM Step for Foundry DSL integration.
//!
//! This module provides `GemmV2Step` which wraps the GEMM compound kernel
//! for use in Foundry model specs.

use std::sync::{Arc, OnceLock};

use metallic_macros::{KernelArgs, MetalStruct};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel, stages::Quantization}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, metals::mma::stages::{GemmEpilogueStage, MmaLoopStage, TileConfig, TileLayoutStage, TileLoadAStage, TileLoadBStage}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

// =============================================================================
// GemmParams - Runtime parameters for GEMM kernel
// =============================================================================

/// Parameters passed to GEMM kernel as a constant buffer.
#[derive(MetalStruct, Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct GemmParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub ldd: i32,
    pub tiles_m: i32,
    pub tiles_n: i32,
    pub gemm_k_iterations: i32,
    pub gemm_k_remainder: i32,
    pub batch_stride_a: i64,
    pub batch_stride_b: i64,
    pub batch_stride_c: i64,
    pub batch_stride_d: i64,
    pub swizzle_log: i32,
    pub batch_ndim: i32,
}

impl GemmParams {
    /// Create params for a simple 2D GEMM (no batching).
    pub fn simple(m: i32, n: i32, k: i32, transpose_a: bool, transpose_b: bool, config: TileConfig) -> Self {
        let (bm, bn, bk, _, _) = config.tile_sizes();
        let bm = bm as i32;
        let bn = bn as i32;
        let bk = bk as i32;

        let params = GemmParams {
            m,
            n,
            k,
            lda: if transpose_a { m } else { k },
            ldb: if transpose_b { k } else { n },
            ldc: n,
            ldd: n,
            tiles_m: (m + bm - 1) / bm,
            tiles_n: (n + bn - 1) / bn,
            gemm_k_iterations: k / bk,
            gemm_k_remainder: k % bk,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_c: 0,
            batch_stride_d: 0,
            swizzle_log: 0, // No swizzle for simple case
            batch_ndim: 0,
        };
        params
    }

    /// Enable tile swizzling for better cache locality.
    pub fn with_swizzle(mut self, log: i32) -> Self {
        self.swizzle_log = log;
        self
    }
}

// =============================================================================
// GemmV2Args - Kernel arguments struct
// =============================================================================

/// Arguments for GEMM kernel dispatch.
#[derive(Debug, KernelArgs)]
pub struct GemmV2Args {
    /// Input activation matrix A [M, K]
    #[arg(buffer = 0)]
    pub a: TensorArg,
    /// Weight matrix B [K, N]
    #[arg(buffer = 1)]
    pub b: TensorArg,
    /// Output matrix D [M, N]
    #[arg(buffer = 2, output)]
    pub d: TensorArg,
    /// Optional C matrix for residual [M, N]
    #[arg(buffer = 3)]
    pub c: TensorArg,
    /// Optional bias [N]
    #[arg(buffer = 4)]
    pub bias: TensorArg,
    /// B scales for quantized weights
    #[arg(buffer = 5)]
    pub b_scales: TensorArg,
    /// Weights per block for quantized B
    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    /// Alpha scaling factor
    #[arg(buffer = 7)]
    pub alpha: f32,
    /// Beta scaling factor
    #[arg(buffer = 8)]
    pub beta: f32,
    /// GEMM parameters struct
    #[arg(buffer = 10)]
    pub params: GemmParams,
}

// =============================================================================
// Kernel Cache - Static cache for compiled GEMM kernels
// =============================================================================

/// Key for kernel cache lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmKernelKey {
    pub a_quant: Quantization,
    pub b_quant: Quantization,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub config: TileConfig,
    pub has_alpha_beta: bool,
    pub has_bias: bool,
}

/// Build and compile a GEMM kernel for the given configuration.
pub fn build_gemm_kernel(key: GemmKernelKey) -> CompiledCompoundKernel {
    let name = format!(
        "gemm_v2_{}_{}_{}{}_{}",
        key.a_quant.short_name(),
        key.b_quant.short_name(),
        if key.transpose_a { "t" } else { "n" },
        if key.transpose_b { "t" } else { "n" },
        match key.config {
            TileConfig::Default => "default",
            TileConfig::SkinnyM => "skinny_m",
            TileConfig::SkinnyN => "skinny_n",
            TileConfig::HighPerformance => "high_perf",
            TileConfig::Custom { .. } => "custom",
        }
    );

    let mut epilogue = GemmEpilogueStage::new();
    if key.has_alpha_beta {
        epilogue = epilogue.with_alpha_beta();
    }
    if key.has_bias {
        epilogue = epilogue.with_bias();
    }

    CompoundKernel::new(&name)
        .prologue(TileLayoutStage::new(key.config, key.transpose_a, key.transpose_b))
        // A loader uses a_quant policy
        .prologue(TileLoadAStage::new(key.a_quant, key.transpose_a))
        // B loader uses b_quant policy (typically where quant happens)
        .prologue(TileLoadBStage::new(key.b_quant, key.transpose_b))
        .main(MmaLoopStage::new().with_k_aligned(false))
        .epilogue(epilogue)
        .with_manual_output(true)
        .compile()
}

/// Get a cached GEMM kernel for the given quantization types.
///
/// This is the unified kernel getter - works for any combination of
/// quantization types (F16, Q8, future Q4, etc).
pub fn get_gemm_kernel(
    a_quant: Quantization,
    b_quant: Quantization,
    transpose_a: bool,
    transpose_b: bool,
    config: TileConfig,
    has_alpha_beta: bool,
    has_bias: bool,
) -> &'static CompiledCompoundKernel {
    // Use a single static cache for all quantization combinations
    static KERNELS: OnceLock<FxHashMap<GemmKernelKey, CompiledCompoundKernel>> = OnceLock::new();

    let key = GemmKernelKey {
        a_quant,
        b_quant,
        transpose_a,
        transpose_b,
        config,
        has_alpha_beta,
        has_bias,
    };

    let map = KERNELS.get_or_init(|| {
        let mut m = FxHashMap::default();

        // Pre-compile common variants
        for a_q in [Quantization::F16] {
            for b_q in [Quantization::F16, Quantization::Q8] {
                for ta in [false] {
                    // A is rarely transposed
                    for tb in [false, true] {
                        for cfg in [
                            TileConfig::Default,
                            TileConfig::SkinnyM,
                            TileConfig::SkinnyN,
                            TileConfig::HighPerformance,
                        ] {
                            // Add variants for alpha/beta/bias as needed
                            for (hab, hb) in [(false, false), (true, false), (false, true), (true, true)] {
                                let k = GemmKernelKey {
                                    a_quant: a_q,
                                    b_quant: b_q,
                                    transpose_a: ta,
                                    transpose_b: tb,
                                    config: cfg,
                                    has_alpha_beta: hab,
                                    has_bias: hb,
                                };
                                m.insert(k, build_gemm_kernel(k));
                            }
                        }
                    }
                }
            }
        }
        m
    });

    map.get(&key).unwrap_or_else(|| {
        panic!(
            "GEMM kernel not pre-compiled for {:?}x{:?} ta={} tb={} config={:?}. \
             Add this combination to the pre-compile list in get_gemm_kernel().",
            a_quant, b_quant, transpose_a, transpose_b, config
        )
    })
}

/// Dispatch configuration for GEMM kernels.
pub fn gemm_dispatch_config(params: &GemmParams, config: TileConfig) -> DispatchConfig {
    let threads_per_tg = config.threads_per_tg() as usize;

    DispatchConfig {
        grid: GridSize::new(params.tiles_n as usize, params.tiles_m as usize, 1),
        group: ThreadgroupSize::new(threads_per_tg, 1, 1),
    }
}

// =============================================================================
// GemmV2Step - DSL Step for GEMM
// =============================================================================

/// GEMM step for Foundry model specs.
///
/// Computes: D = alpha * A @ B + beta * C + bias
///
/// Where:
/// - A: [M, K] activation matrix (F16)
/// - B: [K, N] weight matrix (F16 or quantized)
/// - C: Optional [M, N] matrix for residual
/// - D: [M, N] output matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmV2Step {
    /// Input activation matrix [M, K]
    pub a: Ref,
    /// Weight matrix [K, N] or [N, K] if transposed
    pub b: Ref,
    /// Output matrix [M, N]
    pub output: Ref,
    /// Optional scale buffer for quantized B
    #[serde(default)]
    pub b_scales: Option<Ref>,
    /// M dimension (rows of A)
    pub m_dim: DynamicValue<u32>,
    /// N dimension (cols of B)
    pub n_dim: DynamicValue<u32>,
    /// K dimension (reduction dim)
    pub k_dim: DynamicValue<u32>,
    /// Quantization for B
    #[serde(default)]
    pub b_quant: Quantization,
    /// Transpose A
    #[serde(default)]
    pub transpose_a: bool,
    /// Transpose B
    #[serde(default)]
    pub transpose_b: bool,
    /// Alpha scaling factor
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Beta scaling factor (for residual)
    #[serde(default)]
    pub beta: f32,
    /// Optional bias [N]
    #[serde(default)]
    pub bias: Option<Ref>,
    /// Optional C matrix for residual [M, N]
    #[serde(default)]
    pub c: Option<Ref>,
    /// Weights per block for quantized B
    #[serde(default = "default_wpb")]
    pub weights_per_block: u32,
    /// Tile configuration (None = auto-select)
    #[serde(default)]
    pub tile_config: Option<TileConfig>,
}

fn default_alpha() -> f32 {
    1.0
}
fn default_wpb() -> u32 {
    32
}

#[typetag::serde(name = "GemmV2")]
impl Step for GemmV2Step {
    fn name(&self) -> &'static str {
        "GemmV2"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("GemmV2 only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        // Resolve symbol IDs
        let a_name = bindings.interpolate(self.a.0.clone());
        let b_name = bindings.interpolate(self.b.0.clone());
        let output_name = bindings.interpolate(self.output.0.clone());

        let a_idx = symbols.get_or_create(a_name);
        let b_idx = symbols.get_or_create(b_name.clone());
        let output_idx = symbols.get_or_create(output_name);

        let b_scales_idx = self
            .b_scales
            .as_ref()
            .map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));
        let derived_b_scales_idx = symbols.get_or_create(format!("{b_name}_scales"));
        let bias_idx = self.bias.as_ref().map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));
        let c_idx = self.c.as_ref().map(|r| symbols.get_or_create(bindings.interpolate(r.0.clone())));

        vec![Box::new(CompiledGemmV2Step {
            a_idx,
            b_idx,
            output_idx,
            b_scales_idx,
            derived_b_scales_idx,
            bias_idx,
            c_idx,
            m_dim: self.m_dim.clone(),
            n_dim: self.n_dim.clone(),
            k_dim: self.k_dim.clone(),
            b_quant: self.b_quant,
            transpose_a: self.transpose_a,
            transpose_b: self.transpose_b,
            alpha: self.alpha,
            beta: self.beta,
            weights_per_block: self.weights_per_block,
            tile_config: self.tile_config,
            pipeline: Arc::new(OnceLock::new()),
        })]
    }
}

// =============================================================================
// CompiledGemmV2Step - Executable GEMM step
// =============================================================================

/// Compiled GEMM step ready for execution.
#[derive(Debug, Clone)]
pub struct CompiledGemmV2Step {
    pub a_idx: usize,
    pub b_idx: usize,
    pub output_idx: usize,
    pub b_scales_idx: Option<usize>,
    pub derived_b_scales_idx: usize,
    pub bias_idx: Option<usize>,
    pub c_idx: Option<usize>,
    pub m_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    pub k_dim: DynamicValue<u32>,
    pub b_quant: Quantization,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub alpha: f32,
    pub beta: f32,
    pub weights_per_block: u32,
    pub tile_config: Option<TileConfig>,
    pub pipeline: Arc<OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl CompiledStep for CompiledGemmV2Step {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, bindings: &TensorBindings) -> Result<(), MetalError> {
        // Get tensor args from bindings
        let a = fast_bindings
            .get(self.a_idx)
            .ok_or_else(|| MetalError::InputNotFound("A tensor".into()))?;
        let b = fast_bindings
            .get(self.b_idx)
            .ok_or_else(|| MetalError::InputNotFound("B tensor".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound("output tensor".into()))?;

        // Resolve dimensions
        let m = self.m_dim.resolve(bindings);
        let n = self.n_dim.resolve(bindings);
        let k = self.k_dim.resolve(bindings);

        // Select tile config
        let config = self.tile_config.unwrap_or_else(|| TileConfig::auto_select(m as usize, n as usize));

        // Build params
        let params = GemmParams::simple(m as i32, n as i32, k as i32, self.transpose_a, self.transpose_b, config);
        let dispatch = gemm_dispatch_config(&params, config);

        // Check if we need alpha/beta/bias
        let has_alpha_beta = self.alpha != 1.0 || self.beta != 0.0 || self.c_idx.is_some();
        let has_bias = self.bias_idx.is_some();

        let is_q8 = b.dtype == crate::tensor::Dtype::U8;
        let b_quant = if is_q8 { Quantization::Q8 } else { Quantization::F16 };

        // Get kernel (cached) - uses unified getter for all quant types
        let kernel = get_gemm_kernel(
            Quantization::F16, // A is always F16 (activations)
            b_quant,
            self.transpose_a,
            self.transpose_b,
            config,
            has_alpha_beta,
            has_bias,
        );

        // Build scale arg - Q8 requires real scales (default derived name: "{b}_scales").
        let b_scales = if is_q8 {
            let idx = self.b_scales_idx.unwrap_or(self.derived_b_scales_idx);
            let scales = fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("B scales".into()))?;
            if scales.dtype != crate::tensor::Dtype::U8 {
                return Err(MetalError::InvalidShape(format!(
                    "b_scales must be U8 for Q8 weights (got {:?})",
                    scales.dtype
                )));
            }
            TensorArg::from_tensor(scales)
        } else {
            TensorArg::from_tensor(b) // Dummy for F16 (PolicyF16 ignores scales)
        };

        // Build bias arg - use output as dummy if no bias
        let bias = if let Some(idx) = self.bias_idx {
            let bias_tensor = fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("bias".into()))?;
            TensorArg::from_tensor(bias_tensor)
        } else {
            TensorArg::from_tensor(output) // Dummy
        };

        // Build C arg - use output as dummy if no residual
        let c = if let Some(idx) = self.c_idx {
            let c_tensor = fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("C tensor".into()))?;
            TensorArg::from_tensor(c_tensor)
        } else {
            TensorArg::from_tensor(output) // Dummy
        };

        let args = GemmV2Args {
            a: TensorArg::from_tensor(a),
            b: TensorArg::from_tensor(b),
            d: TensorArg::from_tensor(output),
            c,
            bias,
            b_scales,
            weights_per_block: self.weights_per_block,
            alpha: self.alpha,
            beta: self.beta,
            params,
        };

        // Get pipeline from cache or load it
        let pipeline = if let Some(p) = self.pipeline.get() {
            p
        } else {
            let p = foundry.load_kernel(kernel)?;
            let _ = self.pipeline.set(p);
            self.pipeline.get().unwrap()
        };

        let bound_kernel = kernel.bind(args, dispatch);
        foundry.dispatch_pipeline(pipeline, &bound_kernel, dispatch)
    }

    fn name(&self) -> &'static str {
        "GemmV2"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_params_simple() {
        let params = GemmParams::simple(128, 256, 64, false, false, TileConfig::Default);
        assert_eq!(params.m, 128);
        assert_eq!(params.n, 256);
        assert_eq!(params.k, 64);
        assert_eq!(params.tiles_m, 4); // 128 / 32
        assert_eq!(params.tiles_n, 8); // 256 / 32
        assert_eq!(params.gemm_k_iterations, 4); // 64 / 16
        assert_eq!(params.gemm_k_remainder, 0);
    }

    #[test]
    fn test_gemm_params_unaligned() {
        let params = GemmParams::simple(100, 200, 50, false, false, TileConfig::Default);
        assert_eq!(params.tiles_m, 4); // ceil(100 / 32)
        assert_eq!(params.tiles_n, 7); // ceil(200 / 32)
        assert_eq!(params.gemm_k_iterations, 3); // 50 / 16
        assert_eq!(params.gemm_k_remainder, 2); // 50 % 16
    }
}
