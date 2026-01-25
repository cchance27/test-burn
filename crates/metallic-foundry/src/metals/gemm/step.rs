//! GEMM Step for Foundry DSL integration.
//!
//! This module provides `GemmV2Step` which wraps the GEMM compound kernel
//! for use in Foundry model specs.

use std::sync::Arc;

use metallic_macros::{KernelArgs, MetalStruct};
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, fusion::MetalPolicy, metals::mma::stages::{GemmEpilogueStage, MmaLoopStage, TileConfig, TileLayoutStage, TileLoadAStage, TileLoadBStage}, policy::activation::Activation, spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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

        GemmParams {
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
        }
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GemmKernelKey {
    pub a_quant: String,
    pub b_quant: String,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub config: TileConfig,
    pub has_alpha_beta: bool,
    pub has_bias: bool,
    pub activation: Activation,
}

/// Build and compile a GEMM kernel for the given configuration.
pub fn build_gemm_kernel(key: GemmKernelKey) -> CompiledCompoundKernel {
    let name = format!(
        "gemm_v2_{}_{}_{}{}_{}",
        key.a_quant,
        key.b_quant,
        if key.transpose_a { "t" } else { "n" },
        if key.transpose_b { "t" } else { "n" },
        match key.config {
            TileConfig::Default => "def",
            TileConfig::SkinnyM => "skinny_m",
            TileConfig::SkinnyN => "skinny_n",
            TileConfig::HighPerformance => "perf",
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

    epilogue = epilogue.with_activation(key.activation);

    let a_policy = crate::policy::resolve_policy_by_name(&key.a_quant).expect("Unknown policy A");
    let b_policy = crate::policy::resolve_policy_by_name(&key.b_quant).expect("Unknown policy B");

    CompoundKernel::new(&name)
        .prologue(TileLayoutStage::new(key.config, key.transpose_a, key.transpose_b))
        // A loader uses a_quant policy
        // Coerce Arc<dyn MetalPolicyRuntime> to Arc<dyn MetalPolicy> by creating new Arc or unsafe cast?
        // Safe way: just use parameter. new() expects Arc<dyn MetalPolicy>.
        // MetalPolicyRuntime inherits MetalPolicy.
        .prologue(TileLoadAStage::new(a_policy.clone(), key.transpose_a))
        // B loader uses b_quant policy (typically where quant happens)
        .prologue(TileLoadBStage::new(b_policy.clone(), key.transpose_b))
        .main(MmaLoopStage::new().with_k_aligned(false))
        .epilogue(epilogue)
        .with_manual_output(true)
        .compile()
}

/// Get a cached GEMM kernel for the given configuration.
pub fn get_gemm_kernel(
    a_quant: Arc<dyn MetalPolicy>,
    b_quant: Arc<dyn MetalPolicy>,
    transpose_a: bool,
    transpose_b: bool,
    config: TileConfig,
    has_alpha_beta: bool,
    has_bias: bool,
    activation: Activation,
) -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};

    let gemm_key = GemmKernelKey {
        a_quant: a_quant.short_name().to_string(),
        b_quant: b_quant.short_name().to_string(),
        transpose_a,
        transpose_b,
        config,
        has_alpha_beta,
        has_bias,
        activation,
    };

    let variant = format!("{:?}", gemm_key);
    let key = KernelCacheKey::new("gemm", variant);

    kernel_registry().get_or_build(key, || build_gemm_kernel(gemm_key))
}

/// Dispatch configuration for GEMM kernels.
pub fn gemm_dispatch_config(params: &GemmParams, config: TileConfig) -> DispatchConfig {
    let threads_per_tg = config.threads_per_tg() as usize;

    DispatchConfig {
        grid: GridSize::new(params.tiles_n as usize, params.tiles_m as usize, 1),
        group: ThreadgroupSize::new(threads_per_tg, 1, 1),
    }
}

// Note: GemmV2Step and CompiledGemmV2Step are auto-generated by #[derive(Kernel)] in mod.rs

// Import GemmV2 for execution args construction
use super::GemmV2;

impl CompiledStep for super::CompiledGemmV2Step {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        // Get tensor args from bindings
        let a = fast_bindings
            .get(self.a)
            .ok_or_else(|| MetalError::InputNotFound("A tensor".into()))?;
        let b = fast_bindings
            .get(self.b)
            .ok_or_else(|| MetalError::InputNotFound("B tensor".into()))?;
        let output = fast_bindings
            .get(self.d)
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
        let has_alpha_beta = self.alpha != 1.0 || self.beta != 0.0 || self.c.is_some();

        // Resolve Policy and LoaderStage from the bound weights dtype (do not rely on DSL hints).
        let policy = crate::policy::resolve_policy(b.dtype.into());
        let loader = policy.loader_stage();

        // Bind Weights and Scales using LoaderStage
        // Use implicit fallback for scales
        let b_scales_idx = self.b_scales.or(Some(self.derived_b_scales));

        let b_resolved = crate::spec::ResolvedSymbols {
            weights: self.b,
            scales: b_scales_idx,
            bias: None,
        };

        let loader_args = loader.bind(fast_bindings, &b_resolved);

        let b_arg = loader_args[0].clone();

        // Safety: F16 policy only returns 1 arg (weights). Q8 returns 2 (weights, scales).
        // If args < 2, use b_arg as dummy for scales to satisfy strict struct packing
        let b_scales_arg = if loader_args.len() > 1 {
            loader_args[1].clone()
        } else {
            b_arg.clone()
        };

        // Get kernel (cached) - uses unified getter for all quant types.
        let kernel = get_gemm_kernel(
            std::sync::Arc::new(crate::policy::f16::PolicyF16),
            policy.clone(),
            self.transpose_a,
            self.transpose_b,
            config,
            has_alpha_beta,
            self.bias.is_some(),
            self.activation,
        );

        // Build bias arg - use output as dummy if no bias
        let bias = if let Some(idx) = self.bias {
            let bias_tensor = fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("bias".into()))?;
            TensorArg::from_tensor(bias_tensor)
        } else {
            TensorArg::from_tensor(output) // Dummy
        };

        // Build C arg - use output as dummy if no residual
        let c = if let Some(idx) = self.c {
            let c_tensor = fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("C tensor".into()))?;
            TensorArg::from_tensor(c_tensor)
        } else {
            TensorArg::from_tensor(output) // Dummy
        };

        // Construct Kernel Args using the definition in gemm/mod.rs
        let args = GemmV2 {
            a: TensorArg::from_tensor(a),
            b: b_arg,
            d: TensorArg::from_tensor(output),
            c: Some(c),
            bias: Some(bias),
            b_scales: Some(b_scales_arg),
            weights_per_block: self.weights_per_block,
            alpha: self.alpha,
            beta: self.beta,
            params,
            // Meta fields don't matter for execution binding, but we can fill them or use defaults
            m_dim: self.m_dim.clone(),
            n_dim: self.n_dim.clone(),
            k_dim: self.k_dim.clone(),
            transpose_a: self.transpose_a,
            transpose_b: self.transpose_b,
            tile_config: self.tile_config,
            activation: self.activation,
            derived_b_scales: TensorArg::default(), // Dummy
        };

        foundry.run(&kernel.clone().bind_arc(args, dispatch))
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
