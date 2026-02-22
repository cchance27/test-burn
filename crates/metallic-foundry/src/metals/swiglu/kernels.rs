use std::sync::Arc;

use super::ffn_stages::{FfnDualProjectStage, FfnSwigluWriteStage, FfnWarpReduceStage};
use crate::{
    compound::CompiledCompoundKernel, fusion::MetalPolicy, metals::{
        common::{cache::get_or_build_policy_compound_kernel, composition::manual_output_row_major}, rmsnorm::stages::RmsNormComputeStage
    }
};

pub(super) fn get_fused_ffn_kernel(policy: Arc<dyn MetalPolicy>) -> Arc<CompiledCompoundKernel> {
    get_or_build_policy_compound_kernel("fused_ffn_swiglu", policy, move |policy| {
        let kernel_name = format!("fused_ffn_swiglu_rmsnorm_{}", policy.short_name());

        manual_output_row_major(&kernel_name, 8)
            // Activation input is always F16; quantization only applies to weight loading stages.
            .prologue(RmsNormComputeStage::new(4, 6, 14))
            .main(FfnDualProjectStage::new(policy.clone()).with_norm("inv_rms"))
            .epilogue(FfnWarpReduceStage)
            .epilogue(FfnSwigluWriteStage::new())
            .compile()
    })
}
