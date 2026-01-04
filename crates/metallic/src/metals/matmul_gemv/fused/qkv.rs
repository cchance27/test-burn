//! Fused RMSNorm + QKV Projections.
//!
//! This module contains the fused QKV projection kernel that applies RMSNorm on-the-fly
//! inside the GEMV template, avoiding intermediate buffers and global synchronization.

use std::sync::OnceLock;

use metallic_macros::{GemvKernel, KernelArgs, MetalStruct};
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{DynamicValue, Ref, Step, TensorBindings}
    }, metals::matmul_gemv::{
        hooks::{F16CanonicalHook, RmsnormPrologue}, simd::{GemvEpilogue, GemvHook, GemvStage}
    }, types::{DispatchConfig, GridSize, KernelArg as _, TensorArg, ThreadgroupSize}
};

// ================================================================================================
// Params Struct (MetalStruct generated)
// ================================================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, MetalStruct)]
pub struct QkvFusedParams {
    #[metal(name = "K")]
    pub k: u32,
    #[metal(name = "Nq")]
    pub nq: u32,
    #[metal(name = "Nk")]
    pub nk: u32,
    #[metal(name = "Nv")]
    pub nv: u32,
    pub blocks_per_k: u32,
    pub weights_per_block: u32,
    pub has_bias_q: u32,
    pub has_bias_k: u32,
    pub has_bias_v: u32,
}

// ================================================================================================
// Kernel Args
// ================================================================================================

#[derive(KernelArgs, Clone)]
pub struct QkvF16CanonicalFusedRmsnormArgs {
    pub data_q: TensorArg,
    pub data_k: TensorArg,
    pub data_v: TensorArg,
    pub vector_x: TensorArg,
    #[arg(output)]
    pub out_q: TensorArg,
    #[arg(output)]
    pub out_k: TensorArg,
    #[arg(output)]
    pub out_v: TensorArg,
    pub params: QkvFusedParams,
    pub bias_q: TensorArg,
    pub bias_k: TensorArg,
    pub bias_v: TensorArg,
    pub gamma: TensorArg,
    #[arg(metal_type = "const constant float&")]
    pub epsilon: f32,
}

// ================================================================================================
// SIMD GEMV Config
// ================================================================================================

#[derive(GemvKernel)]
#[gemv_kernel(
    args = "QkvF16CanonicalFusedRmsnormArgs",
    heads = 3,
    cols_per_tg = 8,
    fast_path = true,
    gemv_n0 = "params->Nq",
    data_ptrs("data_q", "data_k", "data_v"),
    result_ptrs("out_q", "out_k", "out_v"),
    n_exprs("params->Nq", "params->Nk", "params->Nv"),
    bias_ptrs("bias_q", "bias_k", "bias_v"),
    has_bias_flags("params->has_bias_q", "params->has_bias_k", "params->has_bias_v"),
    struct_defs_type(QkvFusedParams),
    hook = F16CanonicalHook,
    epilogue = DefaultGemvEpilogue,
)]
struct QkvF16CanonicalFused;

// ================================================================================================
// Default Epilogue (no activation, just bias + output)
// ================================================================================================

struct DefaultGemvEpilogue;
impl DefaultGemvEpilogue {
    const ID: &'static str = "default";
}

impl GemvEpilogue for DefaultGemvEpilogue {
    fn id() -> &'static str {
        Self::ID
    }
}

// ================================================================================================
// Kernel Compilation & Dispatch
// ================================================================================================

fn kernel_name(base: &str, hook: &str, epilogue: &str) -> String {
    format!("{base}__hook_{hook}__epi_{epilogue}")
}

static QKV_KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();

pub fn qkv_f16_canonical_fused_rmsnorm_kernel() -> &'static CompiledCompoundKernel {
    QKV_KERNEL.get_or_init(|| {
        CompoundKernel::new(&kernel_name(
            "foundry_gemv_f16_canonical_qkv_compound",
            <F16CanonicalHook as GemvHook>::id(),
            DefaultGemvEpilogue::ID,
        ))
        .main(GemvStage::<
            RmsnormPrologue,
            QkvF16CanonicalFused,
            F16CanonicalHook,
            DefaultGemvEpilogue,
        >::default())
        .with_manual_output(true)
        .compile()
    })
}

pub fn dispatch_qkv(foundry: &mut Foundry, args: QkvF16CanonicalFusedRmsnormArgs, params: &QkvFusedParams) -> Result<(), MetalError> {
    const TILE_N: usize = 8;
    const TG_WIDTH: usize = 256;
    let max_n = params.nq.max(params.nk).max(params.nv) as usize;
    let grid_x = (max_n + TILE_N - 1) / TILE_N;
    let dispatch = DispatchConfig {
        grid: GridSize::d1(grid_x),
        group: ThreadgroupSize::d1(TG_WIDTH),
    };
    let kernel = qkv_f16_canonical_fused_rmsnorm_kernel().bind(args, dispatch);
    foundry.run(&kernel)
}

// ================================================================================================
// DSL Step
// ================================================================================================

/// Fused RMSNorm + canonical QKV projections (matches legacy `MatmulF16CanonicalQkvFusedRmsnormOp`).
#[derive(Debug, Serialize, Deserialize)]
pub struct QkvF16CanonicalFusedRmsnormStep {
    pub input: Ref,
    pub gamma: Ref,
    pub wq: Ref,
    pub wk: Ref,
    pub wv: Ref,
    pub bq: Ref,
    pub bk: Ref,
    pub bv: Ref,
    pub out_q: Ref,
    pub out_k: Ref,
    pub out_v: Ref,
    #[serde(default)]
    pub epsilon: DynamicValue<f32>,
    #[serde(default)]
    pub weights_per_block: u32,
}

#[typetag::serde(name = "QkvF16CanonicalFusedRmsnorm")]
impl Step for QkvF16CanonicalFusedRmsnormStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let input = bindings.resolve(&self.input)?;
        let gamma = bindings.resolve(&self.gamma)?;
        let wq = bindings.resolve(&self.wq)?;
        let wk = bindings.resolve(&self.wk)?;
        let wv = bindings.resolve(&self.wv)?;
        let bq = bindings.resolve(&self.bq)?;
        let bk = bindings.resolve(&self.bk)?;
        let bv = bindings.resolve(&self.bv)?;
        let out_q = bindings.resolve(&self.out_q)?;
        let out_k = bindings.resolve(&self.out_k)?;
        let out_v = bindings.resolve(&self.out_v)?;

        let eps = self.epsilon.resolve(bindings);
        let wpb = if self.weights_per_block == 0 { 32 } else { self.weights_per_block };

        let k = input.dims().last().copied().unwrap_or(0);
        if k == 0 {
            return Err(MetalError::InvalidShape("QkvF16CanonicalFusedRmsnorm requires non-zero K".into()));
        }
        let nq = out_q.dims().last().copied().unwrap_or(0);
        let nk = out_k.dims().last().copied().unwrap_or(0);
        let nv = out_v.dims().last().copied().unwrap_or(0);
        if nq == 0 || nk == 0 || nv == 0 {
            return Err(MetalError::InvalidShape(format!(
                "QkvF16CanonicalFusedRmsnorm requires non-zero output dims, got nq={nq} nk={nk} nv={nv}"
            )));
        }

        let blocks_per_k = ((k + wpb as usize - 1) / wpb as usize) as u32;

        let params = QkvFusedParams {
            k: k as u32,
            nq: nq as u32,
            nk: nk as u32,
            nv: nv as u32,
            blocks_per_k,
            weights_per_block: wpb,
            has_bias_q: 1,
            has_bias_k: 1,
            has_bias_v: 1,
        };

        let args = QkvF16CanonicalFusedRmsnormArgs {
            data_q: wq,
            data_k: wk,
            data_v: wv,
            vector_x: input,
            out_q,
            out_k,
            out_v,
            params,
            bias_q: bq,
            bias_k: bk,
            bias_v: bv,
            gamma,
            epsilon: eps,
        };
        dispatch_qkv(foundry, args, &params)
    }

    fn name(&self) -> &'static str {
        "QkvF16CanonicalFusedRmsnorm"
    }
}
