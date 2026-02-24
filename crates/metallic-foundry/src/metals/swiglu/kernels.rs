use std::sync::Arc;

use super::ffn_stages::{FfnDualProjectStage, FfnDualProjectUniformStage, FfnSwigluWriteStage, FfnWarpReduceStage};
use crate::{
    compound::{CompiledCompoundKernel, Stage}, fusion::MetalPolicy, metals::{
        common::{cache::get_or_build_compound_kernel, composition::manual_output_row_major, policy_slots::tuple_variant_key}, rmsnorm::stages::RmsNormComputeStage
    }
};

fn compile_fused_ffn_swiglu_kernel(kernel_name: &str, main_stage: Box<dyn Stage>) -> CompiledCompoundKernel {
    manual_output_row_major(kernel_name, 8)
        // Activation input is always F16; quantization only applies to weight loading stages.
        .prologue(RmsNormComputeStage::new(4, 6, 15))
        .main_dyn(main_stage)
        .epilogue(FfnWarpReduceStage)
        .epilogue(FfnSwigluWriteStage::new())
        .compile()
}

pub(super) fn get_fused_ffn_kernel(policy_gate: Arc<dyn MetalPolicy>, policy_up: Arc<dyn MetalPolicy>) -> Arc<CompiledCompoundKernel> {
    if policy_gate.short_name() == policy_up.short_name() {
        let variant = format!("rmsnorm_{}_uniform", policy_gate.short_name());
        let policy = policy_gate.clone();
        return get_or_build_compound_kernel("fused_ffn_swiglu", variant, move || {
            let kernel_name = format!("fused_ffn_swiglu_rmsnorm_{}", policy.short_name());
            let stage = FfnDualProjectUniformStage::new(policy.clone()).with_norm("inv_rms");
            compile_fused_ffn_swiglu_kernel(&kernel_name, Box::new(stage))
        });
    }

    let tuple_key = tuple_variant_key(&[("gate", policy_gate.as_ref()), ("up", policy_up.as_ref())]);
    let variant = format!("rmsnorm_{tuple_key}");
    let gate = policy_gate.clone();
    let up = policy_up.clone();
    get_or_build_compound_kernel("fused_ffn_swiglu", variant, move || {
        let kernel_name = format!("fused_ffn_swiglu_rmsnorm_{}_{}", gate.short_name(), up.short_name());
        let stage = FfnDualProjectStage::new(gate.clone(), up.clone()).with_norm("inv_rms");
        compile_fused_ffn_swiglu_kernel(&kernel_name, Box::new(stage))
    })
}
