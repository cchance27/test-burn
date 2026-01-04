//! Fused RMSNorm + Gate/Up GEMVs + SwiGLU Activation.
//!
//! This module contains the fused SwiGLU kernel that applies RMSNorm on-the-fly,
//! computes gate and up projections simultaneously, and applies SwiGLU activation
//! in the epilogue.

use std::sync::OnceLock;

use metallic_macros::{GemvKernel, KernelArgs, MetalStruct};
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{DynamicValue, Ref, Step, TensorBindings}
    }, metals::{
        matmul_gemv::{
            hooks::{F16CanonicalHook, RmsnormPrologue}, simd::{GemvEpilogue, GemvHook, GemvStage}
        }, swiglu::SwiGluEpilogue
    }, types::{DispatchConfig, GridSize, KernelArg as _, TensorArg, ThreadgroupSize}
};

// ================================================================================================
// Params Struct (MetalStruct generated)
// ================================================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, MetalStruct)]
pub struct Q2FusedParams {
    #[metal(name = "K")]
    pub k: u32,
    #[metal(name = "N0")]
    pub n0: u32,
    #[metal(name = "N1")]
    pub n1: u32,
    pub blocks_per_k: u32,
    pub weights_per_block: u32,
    pub has_bias0: u32,
    pub has_bias1: u32,
}

// ================================================================================================
// Kernel Args
// ================================================================================================

#[derive(KernelArgs, Clone)]
pub struct SwiGluF16CanonicalFusedRmsnormArgs {
    pub data_g: TensorArg,
    pub data_u: TensorArg,
    pub vector_x: TensorArg,
    #[arg(output)]
    pub out_res: TensorArg,
    pub params: Q2FusedParams,
    pub bias_g: TensorArg,
    pub bias_u: TensorArg,
    pub gamma: TensorArg,
    #[arg(metal_type = "const constant float&")]
    pub epsilon: f32,
}

// ================================================================================================
// SIMD GEMV Config
// ================================================================================================

#[derive(GemvKernel)]
#[gemv_kernel(
    args = "SwiGluF16CanonicalFusedRmsnormArgs",
    heads = 2,
    cols_per_tg = 8,
    fast_path = true,
    gemv_n0 = "params->N0",
    data_ptrs("data_g", "data_u"),
    result_ptrs("out_res", "nullptr"),
    n_exprs("params->N0", "params->N1"),
    bias_ptrs("bias_g", "bias_u"),
    has_bias_flags("params->has_bias0", "params->has_bias1"),
    struct_defs_type(Q2FusedParams),
    hook = F16CanonicalHook,
    epilogue = SwiGluEpilogue,
)]
struct SwiGluF16CanonicalFused;

// ================================================================================================
// Kernel Compilation & Dispatch
// ================================================================================================

fn kernel_name(base: &str, hook: &str, epilogue: &str) -> String {
    format!("{base}__hook_{hook}__epi_{epilogue}")
}

static SWIGLU_KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();

pub fn swiglu_f16_canonical_fused_rmsnorm_kernel() -> &'static CompiledCompoundKernel {
    SWIGLU_KERNEL.get_or_init(|| {
        CompoundKernel::new(&kernel_name(
            "foundry_gemv_f16_canonical_swiglu_compound",
            <F16CanonicalHook as GemvHook>::id(),
            <SwiGluEpilogue as GemvEpilogue>::id(),
        ))
        .main(GemvStage::<
            RmsnormPrologue,
            SwiGluF16CanonicalFused,
            F16CanonicalHook,
            SwiGluEpilogue,
        >::default())
        .with_manual_output(true)
        .compile()
    })
}

pub fn dispatch_swiglu(foundry: &mut Foundry, args: SwiGluF16CanonicalFusedRmsnormArgs, params: &Q2FusedParams) -> Result<(), MetalError> {
    const TILE_N: usize = 8;
    const TG_WIDTH: usize = 256;
    let max_n = params.n0.max(params.n1) as usize;
    let grid_x = (max_n + TILE_N - 1) / TILE_N;
    let dispatch = DispatchConfig {
        grid: GridSize::d1(grid_x),
        group: ThreadgroupSize::d1(TG_WIDTH),
    };
    let kernel = swiglu_f16_canonical_fused_rmsnorm_kernel().bind(args, dispatch);
    foundry.run(&kernel)
}

// ================================================================================================
// DSL Step
// ================================================================================================

/// Fused RMSNorm + canonical gate/up GEMVs + SwiGLU activation.
#[derive(Debug, Serialize, Deserialize)]
pub struct SwiGluF16CanonicalFusedRmsnormStep {
    pub input: Ref,
    pub gamma: Ref,
    pub wg: Ref,
    pub wu: Ref,
    pub bg: Ref,
    pub bu: Ref,
    pub out: Ref,
    #[serde(default)]
    pub epsilon: DynamicValue<f32>,
    #[serde(default)]
    pub weights_per_block: u32,
}

#[typetag::serde(name = "SwiGluF16CanonicalFusedRmsnorm")]
impl Step for SwiGluF16CanonicalFusedRmsnormStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let input = bindings.resolve(&self.input)?;
        let gamma = bindings.resolve(&self.gamma)?;
        let wg = bindings.resolve(&self.wg)?;
        let wu = bindings.resolve(&self.wu)?;
        let bg = bindings.resolve(&self.bg)?;
        let bu = bindings.resolve(&self.bu)?;
        let out = bindings.resolve(&self.out)?;

        let eps = self.epsilon.resolve(bindings);
        let wpb = if self.weights_per_block == 0 { 32 } else { self.weights_per_block };

        let k = input.dims().last().copied().unwrap_or(0);
        if k == 0 {
            return Err(MetalError::InvalidShape(
                "SwiGluF16CanonicalFusedRmsnorm requires non-zero K".into(),
            ));
        }
        let n0 = out.dims().last().copied().unwrap_or(0);
        if n0 == 0 {
            return Err(MetalError::InvalidShape(
                "SwiGluF16CanonicalFusedRmsnorm requires non-zero output dim".into(),
            ));
        }

        let blocks_per_k = ((k + wpb as usize - 1) / wpb as usize) as u32;
        let params = Q2FusedParams {
            k: k as u32,
            n0: n0 as u32,
            n1: n0 as u32,
            blocks_per_k,
            weights_per_block: wpb,
            has_bias0: 1,
            has_bias1: 1,
        };

        let args = SwiGluF16CanonicalFusedRmsnormArgs {
            data_g: wg,
            data_u: wu,
            vector_x: input,
            out_res: out,
            params,
            bias_g: bg,
            bias_u: bu,
            gamma,
            epsilon: eps,
        };
        dispatch_swiglu(foundry, args, &params)
    }

    fn name(&self) -> &'static str {
        "SwiGluF16CanonicalFusedRmsnorm"
    }
}
