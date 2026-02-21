use metallic_macros::Stage;

use crate::{
    metals::swiglu::{SwigluParams, SwigluParamsResolved}, policy::activation::Activation, types::TensorArg
};

/// Stage for SwiGLU fused activation.
///
/// Computes: up = SiLU(gate + gate_bias) * (up + up_bias)
///
/// Handles both vectorized and scalar paths based on params.
#[derive(Debug, Clone, Stage)]
#[stage(
    includes("swiglu/swiglu.metal"),
    activation_field = "activation",
    struct_defs_method = "stage_struct_defs",
    emit = r#"
    run_swiglu_stage<{activation_struct}>(gate, up_inout, gate_bias, up_bias, params, gid, lid, tptg);
"#,
    out_var = "swiglu_output"
)]
#[allow(dead_code)]
pub struct SwigluStage {
    #[arg(buffer = 0)]
    gate: TensorArg,
    #[arg(buffer = 1, output)]
    up_inout: TensorArg,
    #[arg(buffer = 2)]
    gate_bias: TensorArg,
    #[arg(buffer = 3)]
    up_bias: TensorArg,
    #[arg(buffer = 4, metal_type = "constant SwigluParams*")]
    params: SwigluParamsResolved,
    #[arg(stage_skip)]
    activation: Activation,
}

impl SwigluStage {
    pub fn new(params: SwigluParamsResolved) -> Self {
        Self {
            gate: TensorArg::default(),
            up_inout: TensorArg::default(),
            gate_bias: TensorArg::default(),
            up_bias: TensorArg::default(),
            params,
            activation: Activation::SiLU,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    fn stage_struct_defs(&self) -> String {
        format!(
            "#define FUSED_KERNEL 1\n#define ACTIVATION {}\n{}",
            self.activation.struct_name(),
            SwigluParams::METAL_STRUCT_DEF
        )
    }
}

#[cfg(test)]
mod tests {
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
}
