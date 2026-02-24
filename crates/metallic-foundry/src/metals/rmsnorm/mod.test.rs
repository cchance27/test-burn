#![cfg(test)]

use super::*;
use crate::{compound::CompoundKernel, metals::rmsnorm::step::RmsNormStandaloneStage};

#[test]
fn test_rmsnorm_params_metal_struct() {
    let def = RmsNormParams::METAL_STRUCT_DEF;
    assert!(def.contains("uint feature_dim;"), "Should have feature_dim: {}", def);
    assert!(def.contains("uint total_elements;"), "Should have total_elements: {}", def);
    assert!(def.contains("float epsilon;"), "Should have epsilon: {}", def);
    assert!(def.contains("struct RmsNormParams"), "Should have struct name: {}", def);
}

#[test]
fn test_rmsnorm_stage_fused() {
    let stage = RmsNormStandaloneStage::default();
    let kernel = CompoundKernel::new("test_fused")
        .main_dyn(Box::new(stage))
        .with_manual_output(true)
        .build();

    let source = kernel.source();

    // Verify buffer arguments collected
    let args = kernel.collect_buffer_args();
    let gamma_arg = args.iter().find(|a| a.name == "gamma").expect("Should have gamma arg");
    assert_eq!(gamma_arg.buffer_index, 3);
    assert_eq!(gamma_arg.metal_type, "const device GammaStorageT*");

    // Verify code emission
    assert!(source.contains("RMSNORM_RUN_CORE_STAGE("));
}
