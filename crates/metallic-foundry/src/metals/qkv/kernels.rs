use std::sync::Arc;

use crate::{
    compound::CompiledCompoundKernel, fusion::MetalPolicy, metals::{
        common::{cache::get_or_build_compound_kernel, composition::manual_output_canonical}, gemv::{GemvStrategy, stages::VectorWidth}, qkv::stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage}, rmsnorm::stages::RmsNormComputeStage
    }
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedQkvKernelVariant {
    NoNorm,
    RmsNorm,
}

impl FusedQkvKernelVariant {
    fn select(has_norm: bool) -> Self {
        if has_norm { Self::RmsNorm } else { Self::NoNorm }
    }

    fn norm_suffix(self) -> &'static str {
        match self {
            Self::NoNorm => "",
            Self::RmsNorm => "_rmsnorm",
        }
    }
}

pub(super) fn get_fused_qkv_kernel(strategy: GemvStrategy, policy: Arc<dyn MetalPolicy>, has_norm: bool) -> Arc<CompiledCompoundKernel> {
    let qkv_variant = FusedQkvKernelVariant::select(has_norm);
    let variant = format!(
        "{:?}_{}_{}",
        strategy,
        policy.short_name(),
        matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm)
    )
    .to_lowercase();

    let policy_clone = policy.clone();
    get_or_build_compound_kernel("fused_qkv", variant, move || {
        let kernel_name = format!("fused_qkv{}_{}", qkv_variant.norm_suffix(), policy_clone.short_name());
        let vec_width = policy_clone.optimization_hints().vector_load_size;

        let mut compound = manual_output_canonical(&kernel_name, 8, vec_width as u32);

        if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
            compound = compound.prologue(RmsNormComputeStage::new(6, 7, 19));
        }

        let vw = match vec_width {
            4 => VectorWidth::Vec4,
            8 => VectorWidth::Vec8,
            // F32-native policies report 16-byte vector loads (float4). Reuse Vec4 lane math.
            16 => VectorWidth::Vec4,
            _ => panic!("Unsupported vector width: {}", vec_width),
        };

        let mut proj = ParallelProjectStage::new(policy_clone.clone()).with_vector_width(vw);
        if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
            proj = proj.with_norm("inv_rms");
        }
        compound = compound.main(proj);

        compound
            .epilogue(MultiWarpReduceStage)
            .epilogue(MultiWriteOutputStage::new())
            .compile()
    })
}
