use std::sync::Arc;

use metallic_macros::KernelArgs;

use super::{
    stages::{
        FlashDecodeFusedStage, FlashDecodeStage, HeadLayoutStage, SdpaParams, SdpaPrefillParams, SdpaPrefillSplitKParams, SdpaPrefillSplitKPartStage, SdpaPrefillSplitKReduceStage, SdpaPrefillStage, SdpaPrefillVariant
    }, variants::FlashDecodeVariant
};
use crate::{
    compound::CompiledCompoundKernel, metals::{
        common::{cache::get_or_build_compound_kernel, composition::manual_output}, rope::{RopeParams, RopeParamsResolved, stage::RopeStage}
    }, types::TensorArg
};

pub(super) fn get_rope_flash_decode_kernel(head_dim: u32, variant: FlashDecodeVariant) -> Arc<CompiledCompoundKernel> {
    let suffix = variant.cache_key_suffix();
    let key_variant = format!("d{}_{}", head_dim, suffix);
    let stage_head_dim = if head_dim == 64 { 64 } else { 128 };
    get_or_build_compound_kernel("rope_flash_decode", key_variant, || {
        let name = "rope_flash_decode_v2";
        let dummy_tensor = TensorArg::default();
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        );
        let dummy_rope = RopeStage::new(dummy_tensor.clone(), dummy_tensor.clone(), RopeParams::default());
        let dummy_core: Box<dyn crate::compound::Stage> = match stage_head_dim {
            128 => Box::new(FlashDecodeFusedStage::<128>::new(SdpaParams::default(), variant)),
            64 => Box::new(FlashDecodeFusedStage::<64>::new(SdpaParams::default(), variant)),
            _ => panic!("Unsupported decode stage_head_dim for dummy core: {}", stage_head_dim),
        };

        manual_output(name)
            .prologue(dummy_layout)
            .prologue(dummy_rope)
            .main_dyn(dummy_core)
            .compile()
    })
}

pub(super) fn get_flash_decode_kernel(head_dim: u32, variant: FlashDecodeVariant) -> Arc<CompiledCompoundKernel> {
    let suffix = variant.cache_key_suffix();
    let stage_head_dim = if head_dim == 64 { 64 } else { 128 };
    let name_suffix = format!("d{}_h{}_{}", stage_head_dim, head_dim, suffix);
    get_or_build_compound_kernel("flash_decode_standalone", name_suffix.clone(), || {
        let dummy_tensor = TensorArg::default();
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        );

        let stage_box: Box<dyn crate::compound::Stage> = match stage_head_dim {
            128 => Box::new(FlashDecodeStage::<128>::new(SdpaParams::default(), variant)),
            64 => Box::new(FlashDecodeStage::<64>::new(SdpaParams::default(), variant)),
            _ => panic!("Unsupported decode stage_head_dim for Flash Decode: {}", stage_head_dim),
        };

        manual_output(&format!("flash_decode_standalone_{}", name_suffix))
            .prologue(dummy_layout)
            .main_dyn(stage_box)
            .compile()
    })
}

pub(super) fn get_sdpa_prefill_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    let variant = format!("w{}", prefill_warps);
    get_or_build_compound_kernel("sdpa_prefill", variant, || {
        let mut stage = SdpaPrefillStage::new(SdpaPrefillParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        manual_output(&format!("sdpa_prefill_w{}", prefill_warps)).main(stage).compile()
    })
}

pub(super) fn get_sdpa_prefill_splitk_part_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    let variant = format!("w{}", prefill_warps);
    get_or_build_compound_kernel("sdpa_prefill_splitk_part", variant, || {
        let mut stage = SdpaPrefillSplitKPartStage::new(SdpaPrefillSplitKParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        manual_output(&format!("sdpa_prefill_splitk_part_w{}", prefill_warps))
            .main(stage)
            .compile()
    })
}

pub(super) fn get_sdpa_prefill_splitk_reduce_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    let variant = format!("w{}", prefill_warps);
    get_or_build_compound_kernel("sdpa_prefill_splitk_reduce", variant, || {
        let mut stage = SdpaPrefillSplitKReduceStage::new(SdpaPrefillSplitKParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        manual_output(&format!("sdpa_prefill_splitk_reduce_w{}", prefill_warps))
            .main(stage)
            .compile()
    })
}

/// Arguments for the fused RoPE→SDPA compound kernel.
///
/// Buffer binding order must match the stage sequence exactly:
/// `HeadLayoutStage` → `RopeStage` → `FlashDecodeFusedStage*`.
#[derive(KernelArgs)]
pub(super) struct RopeFlashDecodeArgs {
    #[arg(metal_type = "const device InputStorageT*")]
    pub q: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub k: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub v: TensorArg,
    #[arg(output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    pub q_stride_b: u32,
    pub q_stride_h: u32,
    pub k_stride_b: u32,
    pub k_stride_h: u32,
    pub v_stride_b: u32,
    pub v_stride_h: u32,
    pub out_stride_b: u32,
    pub out_stride_h: u32,
    #[arg(metal_type = "const device TensorStorageT*")]
    pub cos: TensorArg,
    #[arg(metal_type = "const device TensorStorageT*")]
    pub sin: TensorArg,
    pub params_rope: RopeParamsResolved,
    pub sdpa_params: SdpaParams,
}

/// Arguments for standalone Flash Decode (no RoPE)
#[derive(Debug, KernelArgs)]
pub(super) struct FlashDecodeArgs {
    #[arg(metal_type = "const device InputStorageT*")]
    pub q: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub k: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub v: TensorArg,
    #[arg(output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    pub q_stride_b: u32,
    pub q_stride_h: u32,
    pub k_stride_b: u32,
    pub k_stride_h: u32,
    pub v_stride_b: u32,
    pub v_stride_h: u32,
    pub out_stride_b: u32,
    pub out_stride_h: u32,
    pub sdpa_params: SdpaParams,
}

#[derive(Debug, KernelArgs)]
pub(super) struct SdpaPrefillArgs {
    #[arg(metal_type = "const device InputStorageT*")]
    pub q: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub k: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub v: TensorArg,
    #[arg(output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    pub params: SdpaPrefillParams,
}

#[derive(Debug, KernelArgs)]
pub(super) struct SdpaPrefillSplitKPartArgs {
    #[arg(metal_type = "const device InputStorageT*")]
    pub q: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub k: TensorArg,
    #[arg(metal_type = "const device InputStorageT*")]
    pub v: TensorArg,
    #[arg(metal_type = "device float*")]
    pub partial_acc: TensorArg,
    #[arg(metal_type = "device float*")]
    pub partial_m: TensorArg,
    #[arg(metal_type = "device float*")]
    pub partial_l: TensorArg,
    pub params: SdpaPrefillSplitKParams,
}

#[derive(Debug, KernelArgs)]
pub(super) struct SdpaPrefillSplitKReduceArgs {
    #[arg(output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(metal_type = "const device float*")]
    pub partial_acc: TensorArg,
    #[arg(metal_type = "const device float*")]
    pub partial_m: TensorArg,
    #[arg(metal_type = "const device float*")]
    pub partial_l: TensorArg,
    pub params: SdpaPrefillSplitKParams,
}
