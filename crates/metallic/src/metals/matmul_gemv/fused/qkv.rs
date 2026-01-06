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

    fn compile(
        &self,
        resolver: &mut TensorBindings,
        symbols: &mut crate::foundry::spec::SymbolTable,
    ) -> Vec<Box<dyn crate::foundry::spec::CompiledStep>> {
        let input_idx = symbols.get_or_create(resolver.interpolate(self.input.0.clone()));
        let gamma_idx = symbols.get_or_create(resolver.interpolate(self.gamma.0.clone()));
        let wq_idx = symbols.get_or_create(resolver.interpolate(self.wq.0.clone()));
        let wk_idx = symbols.get_or_create(resolver.interpolate(self.wk.0.clone()));
        let wv_idx = symbols.get_or_create(resolver.interpolate(self.wv.0.clone()));
        let bq_idx = symbols.get_or_create(resolver.interpolate(self.bq.0.clone()));
        let bk_idx = symbols.get_or_create(resolver.interpolate(self.bk.0.clone()));
        let bv_idx = symbols.get_or_create(resolver.interpolate(self.bv.0.clone()));
        let out_q_idx = symbols.get_or_create(resolver.interpolate(self.out_q.0.clone()));
        let out_k_idx = symbols.get_or_create(resolver.interpolate(self.out_k.0.clone()));
        let out_v_idx = symbols.get_or_create(resolver.interpolate(self.out_v.0.clone()));

        vec![Box::new(CompiledQkvF16CanonicalFusedRmsnormStep {
            input_idx,
            gamma_idx,
            wq_idx,
            wk_idx,
            wv_idx,
            bq_idx,
            bk_idx,
            bv_idx,
            out_q_idx,
            out_k_idx,
            out_v_idx,
            epsilon: self.epsilon.clone(),
            weights_per_block: self.weights_per_block,
        })]
    }
}

#[derive(Debug)]
pub struct CompiledQkvF16CanonicalFusedRmsnormStep {
    pub input_idx: usize,
    pub gamma_idx: usize,
    pub wq_idx: usize,
    pub wk_idx: usize,
    pub wv_idx: usize,
    pub bq_idx: usize,
    pub bk_idx: usize,
    pub bv_idx: usize,
    pub out_q_idx: usize,
    pub out_k_idx: usize,
    pub out_v_idx: usize,
    pub epsilon: DynamicValue<f32>,
    pub weights_per_block: u32,
}

impl crate::foundry::spec::CompiledStep for CompiledQkvF16CanonicalFusedRmsnormStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let input = fast_bindings
            .get(self.input_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Input tensor not found at idx {}", self.input_idx)))?;
        let gamma = fast_bindings
            .get(self.gamma_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Gamma tensor not found at idx {}", self.gamma_idx)))?;
        let wq = fast_bindings
            .get(self.wq_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Wq tensor not found at idx {}", self.wq_idx)))?;
        let wk = fast_bindings
            .get(self.wk_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Wk tensor not found at idx {}", self.wk_idx)))?;
        let wv = fast_bindings
            .get(self.wv_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Wv tensor not found at idx {}", self.wv_idx)))?;
        let bq = fast_bindings
            .get(self.bq_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Bq tensor not found at idx {}", self.bq_idx)))?;
        let bk = fast_bindings
            .get(self.bk_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Bk tensor not found at idx {}", self.bk_idx)))?;
        let bv = fast_bindings
            .get(self.bv_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Bv tensor not found at idx {}", self.bv_idx)))?;
        let out_q = fast_bindings
            .get(self.out_q_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("OutQ tensor not found at idx {}", self.out_q_idx)))?;
        let out_k = fast_bindings
            .get(self.out_k_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("OutK tensor not found at idx {}", self.out_k_idx)))?;
        let out_v = fast_bindings
            .get(self.out_v_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("OutV tensor not found at idx {}", self.out_v_idx)))?;

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
            data_q: wq.clone(),
            data_k: wk.clone(),
            data_v: wv.clone(),
            vector_x: input.clone(),
            out_q: out_q.clone(),
            out_k: out_k.clone(),
            out_v: out_v.clone(),
            params,
            bias_q: bq.clone(),
            bias_k: bk.clone(),
            bias_v: bv.clone(),
            gamma: gamma.clone(),
            epsilon: eps,
        };
        dispatch_qkv(foundry, args, &params)
    }
}
