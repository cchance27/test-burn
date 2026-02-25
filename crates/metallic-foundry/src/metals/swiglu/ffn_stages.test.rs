#![cfg(test)]

use std::sync::Arc;

use super::{FfnDualProjectStage, FfnSwigluWriteStage};
use crate::{
    compound::Stage, policy::{activation::Activation, f16::PolicyF16}
};

#[test]
fn dual_project_emit_and_policy_metadata() {
    let stage = FfnDualProjectStage::new(Arc::new(PolicyF16), Arc::new(PolicyF16)).with_norm("inv_rms");
    let (out, code) = stage.emit("unused");

    assert_eq!(out, "gu_partial");
    assert!(code.contains("run_ffn_dual_project_stage<PolicyF16, 8, true>"));
    assert!(code.contains("gamma, inv_rms"));
    assert_eq!(stage.policy_meta().map(|m| m.struct_name), Some("PolicyF16"));
}

#[test]
fn swiglu_write_emit_and_activation_metadata() {
    let stage = FfnSwigluWriteStage::new().with_activation(Activation::ReLU);
    let includes = stage.includes();
    let (_, code) = stage.emit("gu_final");

    assert_eq!(includes, vec!["swiglu/swiglu.metal", "policies/activations.metal"]);
    assert!(code.contains("run_swiglu_write_stage<ActivationReLU>(gu_final"));
    assert_eq!(stage.activation_meta(), Some(Activation::ReLU));
}
