use metallic_foundry::{
    compound::Stage, metals::{
        attention::stages::{KvTileConfig, KvTileLayoutStage}, flashattention::{
            stages::{FlashDecodeStage, SdpaParams}, step::RopeFlashDecodeStep, variants::select_flash_decode_variant_m2m3
        }
    }
};

#[test]
fn flashattention_includes_flash_decode_metal() {
    let stage = FlashDecodeStage::<64>::new(SdpaParams::default(), select_flash_decode_variant_m2m3(64, 1024));
    let includes = stage.includes();
    assert!(
        includes.contains(&"flashattention/decode_kernels.metal"),
        "expected flash decode include, got: {includes:?}"
    );
}

#[test]
fn flashattention_source_mentions_streaming_softmax() {
    let src = RopeFlashDecodeStep::source();
    assert!(
        src.contains("run_flash_decode_fused_half2_stage") || src.contains("run_flash_decode_fused_half4_stage"),
        "expected flash decode function in generated source"
    );
    assert!(!src.contains("sdpa_decode_vectorized"), "legacy decode symbol unexpectedly present");
}

#[test]
fn sdpa_standalone_stage_uses_expected_sdpa_params_buffer_slot() {
    let stage = FlashDecodeStage::<64>::new(SdpaParams::default(), select_flash_decode_variant_m2m3(64, 1024));
    let args = stage.buffer_args();
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].name, "sdpa_params");
    assert_eq!(args[0].buffer_index, 12);
}

#[test]
fn kv_tile_layout_stage_defines_bn_constant() {
    let stage = KvTileLayoutStage::new(KvTileConfig::new(64));
    let defs = stage.struct_defs();
    assert!(defs.contains("#define KV_TILE_BN 64"));
}
