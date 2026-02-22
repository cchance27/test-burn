#![cfg(test)]

use super::SwigluStage;
use crate::{compound::Stage, metals::swiglu::SwigluParamsResolved, policy::activation::Activation};

#[test]
fn swiglu_stage_derived_emit_and_activation_meta() {
    let params = SwigluParamsResolved {
        total_elements: 8,
        bias_len: 4,
        vector_width: 4,
        gate_leading_stride: 4,
        up_leading_stride: 4,
    };
    let stage = SwigluStage::new(params).with_activation(Activation::GELU);
    let (out, code) = stage.emit("unused");

    assert_eq!(out, "swiglu_output");
    assert!(code.contains("run_swiglu_stage<ActivationGELU>("));
    assert_eq!(stage.activation_meta(), Some(Activation::GELU));
}
