//! GEMM Step for Foundry DSL integration.
//!
//! This module provides `GemmV2Step` which wraps the GEMM compound kernel
//! for use in Foundry model specs.

use std::sync::Arc;

use metallic_macros::{KernelArgs, MetalStruct};
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::CompiledCompoundKernel, fusion::MetalPolicy, metals::{
        common::{cache::get_or_build_compound_kernel, composition::manual_output, dtype_contract::KernelDtypeDescriptor}, mma::stages::{GemmEpilogueStage, MmaLoopStage, TileConfig, TileLayoutStage, TileLoadAStage, TileLoadBStage}
    }, policy::activation::Activation, spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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
    #[arg(buffer = 0, metal_type = "const device InputStorageT*")]
    pub a: TensorArg,
    /// Weight matrix B [K, N]
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub b: TensorArg,
    /// Output matrix D [M, N]
    #[arg(buffer = 2, output, metal_type = "device OutputStorageT*")]
    pub d: TensorArg,
    /// Optional C matrix for residual [M, N]
    #[arg(buffer = 3, metal_type = "const device ResidualStorageT*")]
    pub c: TensorArg,
    /// Optional bias [N]
    #[arg(buffer = 4, metal_type = "const device BiasStorageT*")]
    pub bias: TensorArg,
    /// B scales for quantized weights
    #[arg(buffer = 5, metal_type = "const device uchar*")]
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
    /// Whether B buffer is canonical packed block-quant layout (1D packed stream)
    #[arg(buffer = 9)]
    pub b_is_canonical: u32,
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

#[inline]
fn tile_config_suffix(config: TileConfig) -> &'static str {
    match config {
        TileConfig::Default => "def",
        TileConfig::SkinnyM => "skinny_m",
        TileConfig::SkinnyN => "skinny_n",
        TileConfig::HighPerformance => "perf",
        TileConfig::Custom { .. } => "custom",
    }
}

#[inline]
fn transpose_suffix(transpose: bool) -> &'static str {
    if transpose { "t" } else { "n" }
}

fn gemm_kernel_name(key: &GemmKernelKey) -> String {
    format!(
        "gemm_v2_{}_{}_{}{}_{}",
        key.a_quant,
        key.b_quant,
        transpose_suffix(key.transpose_a),
        transpose_suffix(key.transpose_b),
        tile_config_suffix(key.config)
    )
}

fn gemm_variant_key(key: &GemmKernelKey) -> String {
    let policy_tuple = format!("a={}_b={}", key.a_quant, key.b_quant);
    let transpose = format!("{}{}", transpose_suffix(key.transpose_a), transpose_suffix(key.transpose_b));
    let tile = tile_config_suffix(key.config);
    let alpha_beta = if key.has_alpha_beta { "ab1" } else { "ab0" };
    let bias = if key.has_bias { "bias1" } else { "bias0" };
    format!(
        "{}_{}_{}_{}_{}_act{}",
        policy_tuple,
        transpose,
        tile,
        alpha_beta,
        bias,
        key.activation.struct_name()
    )
}

fn build_gemm_compound(
    name: &str,
    key: &GemmKernelKey,
    a_policy: Arc<dyn MetalPolicy>,
    b_policy: Arc<dyn MetalPolicy>,
) -> CompiledCompoundKernel {
    let mut epilogue = GemmEpilogueStage::new();
    if key.has_alpha_beta {
        epilogue = epilogue.with_alpha_beta();
    }
    if key.has_bias {
        epilogue = epilogue.with_bias();
    }
    epilogue = epilogue.with_activation(key.activation);

    manual_output(name)
        .prologue(TileLayoutStage::new(key.config, key.transpose_a, key.transpose_b))
        .prologue(TileLoadAStage::new(a_policy, key.transpose_a))
        .prologue(TileLoadBStage::new(b_policy, key.transpose_b))
        .main(MmaLoopStage::new().with_k_aligned(false))
        .epilogue(epilogue)
        .compile()
}

/// Build and compile a GEMM kernel for the given configuration.
pub fn build_gemm_kernel(key: GemmKernelKey) -> CompiledCompoundKernel {
    let name = gemm_kernel_name(&key);
    let a_policy = crate::policy::resolve_policy_by_name(&key.a_quant).expect("Unknown policy A");
    let b_policy = crate::policy::resolve_policy_by_name(&key.b_quant).expect("Unknown policy B");
    build_gemm_compound(&name, &key, a_policy, b_policy)
}

/// Get a cached GEMM kernel for the given configuration.
#[allow(clippy::too_many_arguments)]
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

    let variant = gemm_variant_key(&gemm_key);
    get_or_build_compound_kernel("gemm", variant, || build_gemm_kernel(gemm_key))
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
        let c_tensor = if let Some(idx) = self.c {
            Some(fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("C tensor".into()))?)
        } else {
            None
        };
        let bias_tensor = if let Some(idx) = self.bias {
            Some(fast_bindings.get(idx).ok_or_else(|| MetalError::InputNotFound("bias".into()))?)
        } else {
            None
        };

        // Resolve dimensions
        let m = self.m_dim.resolve(bindings);
        let n = self.n_dim.resolve(bindings);
        let k = self.k_dim.resolve(bindings);

        // Select tile config
        let config = self.tile_config.unwrap_or_else(|| {
            let storage_bytes = KernelDtypeDescriptor::from_source_dtype(output.dtype)
                .map(|desc| desc.storage_size_bytes)
                .unwrap_or_else(|_| output.dtype.size_bytes());
            TileConfig::auto_select_for_storage(m as usize, n as usize, storage_bytes)
        });

        // Build params
        let params = GemmParams::simple(m as i32, n as i32, k as i32, self.transpose_a, self.transpose_b, config);
        let dispatch = gemm_dispatch_config(&params, config);

        // Check if we need alpha/beta/bias
        let has_alpha_beta = self.alpha != 1.0 || self.beta != 0.0 || self.c.is_some();

        // Resolve policies from runtime tensor dtypes (no DSL hints).
        let a_policy = crate::policy::resolve_policy(a.dtype);
        let policy = crate::policy::resolve_policy(b.dtype);
        let out_policy = crate::policy::resolve_policy(output.dtype);

        // Activations/output/residual/bias must be dense (non-block-quant) tensors.
        if a_policy.has_scale() {
            return Err(MetalError::OperationNotSupported(format!(
                "GemmV2 does not support quantized activation tensor dtype {:?}.",
                a.dtype
            )));
        }
        if out_policy.has_scale() {
            return Err(MetalError::OperationNotSupported(format!(
                "GemmV2 does not support quantized output tensor dtype {:?}.",
                output.dtype
            )));
        }
        if let Some(c_ref) = c_tensor {
            let c_policy = crate::policy::resolve_policy(c_ref.dtype);
            if c_policy.has_scale() {
                return Err(MetalError::OperationNotSupported(format!(
                    "GemmV2 does not support quantized residual tensor dtype {:?}.",
                    c_ref.dtype
                )));
            }
        }
        if let Some(bias_ref) = bias_tensor {
            let bias_policy = crate::policy::resolve_policy(bias_ref.dtype);
            if bias_policy.has_scale() {
                return Err(MetalError::OperationNotSupported(format!(
                    "GemmV2 does not support quantized bias tensor dtype {:?}.",
                    bias_ref.dtype
                )));
            }
        }

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
        let b_is_canonical = if b_arg.dims.len() == 1 { 1u32 } else { 0u32 };

        // Safety: F16 policy only returns 1 arg (weights). Q8 returns 2 (weights, scales).
        // If args < 2, use b_arg as dummy for scales to satisfy strict struct packing
        let b_scales_arg = if loader_args.len() > 1 {
            loader_args[1].clone()
        } else {
            b_arg.clone()
        };
        let weights_per_block = if policy.has_scale() {
            policy.meta().weights_per_block as u32
        } else {
            self.weights_per_block
        };

        // Get kernel (cached) - uses unified getter for all quant types.
        let kernel = get_gemm_kernel(
            a_policy.clone(),
            policy.clone(),
            self.transpose_a,
            self.transpose_b,
            config,
            has_alpha_beta,
            self.bias.is_some(),
            self.activation,
        );

        // Build bias arg - use output as dummy if no bias
        let bias = if let Some(bias_ref) = bias_tensor {
            TensorArg::from_tensor(bias_ref)
        } else {
            TensorArg::from_tensor(output) // Dummy
        };

        // Build C arg - use output as dummy if no residual
        let c = if let Some(c_ref) = c_tensor {
            TensorArg::from_tensor(c_ref)
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
            weights_per_block,
            alpha: self.alpha,
            beta: self.beta,
            b_is_canonical,
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

#[path = "step.test.rs"]
mod tests;
