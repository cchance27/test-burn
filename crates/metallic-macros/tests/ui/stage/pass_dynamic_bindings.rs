use std::sync::Arc;

use metallic_macros::Stage;
use metallic_foundry::{
    compound::{BufferArg, Stage as StageTrait},
    fusion::MetalPolicy,
    policy::{activation::Activation, f16::PolicyF16},
};

#[derive(Stage)]
#[stage(
    includes("swiglu/swiglu.metal"),
    activation_field = "activation",
    policy_field = "policy",
    buffer_args_fn = "stage_buffer_args",
    template_bindings(
        width = "self.width",
        stage_name = "\"demo\""
    ),
    emit = r#"
    // {stage_name}
    float {out_var} = (float){width};
    run_swiglu_write_stage<{activation_struct}>(float2({input_var}, {input_var}), output, bias, bias, 0u, 0u, lane_id, row_idx, batch_idx, n_dim);
    // {policy_struct}
"#,
    out_var = "demo_out"
)]
struct DynamicStage {
    activation: Activation,
    policy: Arc<dyn MetalPolicy>,
    width: u32,
}

impl DynamicStage {
    fn stage_buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "bias",
                metal_type: "const device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "n_dim",
                metal_type: "constant uint&",
                buffer_index: 2,
            },
        ]
    }
}

fn main() {
    let s = DynamicStage {
        activation: Activation::SiLU,
        policy: Arc::new(PolicyF16),
        width: 8,
    };
    let _ = s.includes();
    let _ = s.buffer_args();
    let _ = s.emit("x");
    let _ = s.activation_meta();
    let _ = s.policy_meta();
}

