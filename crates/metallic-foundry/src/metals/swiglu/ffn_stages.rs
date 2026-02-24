//! Fused FFN stages: RMSNorm + Gate/Up projections + SwiGLU writeback.

use std::sync::Arc;

use metallic_macros::Stage;

use crate::{
    fusion::MetalPolicy, metals::{gemv::stages::VectorWidth, swiglu::SwigluParams}, policy::activation::Activation, types::TensorArg
};

/// Dual projection stage for FFN (gate + up) with optional RMSNorm fusion.
#[derive(Debug, Clone, Stage)]
#[stage(
    includes(
        "gemv/common.metal",
        "gemv/dot.metal",
        "gemv/vectorized_stage.metal",
        "gemv/scalar_output.metal",
        "swiglu/ffn_stages.metal"
    ),
    include_exprs("self.policy_gate.header()", "self.policy_up.header()"),
    template_bindings(
        policy_gate_struct = "self.policy_gate.struct_name()",
        policy_up_struct = "self.policy_up.struct_name()",
        vec_width = "self.vector_width.elements()",
        has_norm = "if self.norm_shared_name.is_some() { \"true\" } else { \"false\" }",
        norm_var = "self.norm_shared_name.as_deref().unwrap_or(\"0.0f\")"
    ),
    emit = r#"
    float2 {out_var} = run_ffn_dual_project_stage<{policy_gate_struct}, {policy_up_struct}, {vec_width}, {has_norm}>(
        w_gate,
        s_gate,
        w_up,
        s_up,
        input,
        k_dim,
        n_dim,
        weights_per_block_gate,
        weights_per_block_up,
        gamma,
        {norm_var},
        lane_id,
        row_idx,
        batch_idx
    );
"#,
    out_var = "gu_partial"
)]
// DEBT: fields are consumed by `#[derive(Stage)]` codegen and Metal emission, not direct Rust reads.
#[allow(dead_code)]
pub struct FfnDualProjectStage {
    #[arg(buffer = 0)]
    #[arg(metal_type = "const device uchar*")]
    w_gate: TensorArg,
    #[arg(buffer = 1)]
    #[arg(metal_type = "const device uchar*")]
    s_gate: TensorArg,
    #[arg(buffer = 2)]
    #[arg(metal_type = "const device uchar*")]
    w_up: TensorArg,
    #[arg(buffer = 3)]
    #[arg(metal_type = "const device uchar*")]
    s_up: TensorArg,
    #[arg(buffer = 4, metal_type = "const device InputStorageT*")]
    input: TensorArg,
    #[arg(buffer = 6)]
    k_dim: u32,
    #[arg(buffer = 7)]
    n_dim: u32,
    #[arg(buffer = 8)]
    weights_per_block_gate: u32,
    #[arg(buffer = 9)]
    weights_per_block_up: u32,
    #[arg(buffer = 10, metal_type = "const device GammaStorageT*")]
    gamma: TensorArg,
    #[arg(stage_skip)]
    policy_gate: Arc<dyn MetalPolicy>,
    #[arg(stage_skip)]
    policy_up: Arc<dyn MetalPolicy>,
    #[arg(stage_skip)]
    vector_width: VectorWidth,
    #[arg(stage_skip)]
    norm_shared_name: Option<String>,
}

impl FfnDualProjectStage {
    pub fn new(policy_gate: Arc<dyn MetalPolicy>, policy_up: Arc<dyn MetalPolicy>) -> Self {
        Self {
            w_gate: TensorArg::default(),
            s_gate: TensorArg::default(),
            w_up: TensorArg::default(),
            s_up: TensorArg::default(),
            input: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            weights_per_block_gate: 0,
            weights_per_block_up: 0,
            gamma: TensorArg::default(),
            policy_gate,
            policy_up,
            vector_width: VectorWidth::Vec8,
            norm_shared_name: None,
        }
    }

    pub fn with_norm(mut self, shared_name: &str) -> Self {
        self.norm_shared_name = Some(shared_name.to_string());
        self
    }
}

/// Uniform-policy projection stage preserving the single-policy FFN path.
#[derive(Debug, Clone, Stage)]
#[stage(
    includes(
        "gemv/common.metal",
        "gemv/dot.metal",
        "gemv/vectorized_stage.metal",
        "gemv/scalar_output.metal",
        "swiglu/ffn_stages.metal"
    ),
    include_exprs("self.policy.header()"),
    template_bindings(
        policy_struct = "self.policy.struct_name()",
        vec_width = "self.vector_width.elements()",
        has_norm = "if self.norm_shared_name.is_some() { \"true\" } else { \"false\" }",
        norm_var = "self.norm_shared_name.as_deref().unwrap_or(\"0.0f\")"
    ),
    emit = r#"
    float2 {out_var} = run_ffn_dual_project_stage_uniform<{policy_struct}, {vec_width}, {has_norm}>(
        w_gate, s_gate, w_up, s_up, input, k_dim, n_dim, weights_per_block_gate, gamma, {norm_var}, lane_id, row_idx, batch_idx
    );
"#,
    out_var = "gu_partial"
)]
// DEBT: fields are consumed by `#[derive(Stage)]` codegen and Metal emission, not direct Rust reads.
#[allow(dead_code)]
pub struct FfnDualProjectUniformStage {
    #[arg(buffer = 0)]
    #[arg(metal_type = "const device uchar*")]
    w_gate: TensorArg,
    #[arg(buffer = 1)]
    #[arg(metal_type = "const device uchar*")]
    s_gate: TensorArg,
    #[arg(buffer = 2)]
    #[arg(metal_type = "const device uchar*")]
    w_up: TensorArg,
    #[arg(buffer = 3)]
    #[arg(metal_type = "const device uchar*")]
    s_up: TensorArg,
    #[arg(buffer = 4, metal_type = "const device InputStorageT*")]
    input: TensorArg,
    #[arg(buffer = 6)]
    k_dim: u32,
    #[arg(buffer = 7)]
    n_dim: u32,
    #[arg(buffer = 8)]
    weights_per_block_gate: u32,
    #[arg(buffer = 10, metal_type = "const device GammaStorageT*")]
    gamma: TensorArg,
    #[arg(stage_skip)]
    policy: Arc<dyn MetalPolicy>,
    #[arg(stage_skip)]
    vector_width: VectorWidth,
    #[arg(stage_skip)]
    norm_shared_name: Option<String>,
}

impl FfnDualProjectUniformStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            w_gate: TensorArg::default(),
            s_gate: TensorArg::default(),
            w_up: TensorArg::default(),
            s_up: TensorArg::default(),
            input: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            weights_per_block_gate: 0,
            gamma: TensorArg::default(),
            policy,
            vector_width: VectorWidth::Vec8,
            norm_shared_name: None,
        }
    }

    pub fn with_norm(mut self, shared_name: &str) -> Self {
        self.norm_shared_name = Some(shared_name.to_string());
        self
    }
}

/// Warp reduction stage for float2 accumulators.
#[derive(Debug, Clone, Default, Stage)]
#[stage(
    emit = r#"
    float2 gu_sum = {input_var};
    for (uint offset = 16; offset > 0; offset /= 2) {
        gu_sum.x += simd_shuffle_down(gu_sum.x, offset);
        gu_sum.y += simd_shuffle_down(gu_sum.y, offset);
    }
    float2 {out_var} = gu_sum;
"#,
    out_var = "gu_final"
)]
pub struct FfnWarpReduceStage;

/// Write stage for SwiGLU output.
#[derive(Debug, Clone, Stage)]
#[stage(
    includes("swiglu/swiglu.metal"),
    activation_field = "activation",
    struct_defs_method = "stage_struct_defs",
    emit = r#"
    run_swiglu_write_stage<{activation_struct}>({input_var}, output, b_gate, b_up, has_b_gate, has_b_up, lane_id, row_idx, batch_idx, n_dim);
"#,
    out_var = "void"
)]
// DEBT: fields are consumed by `#[derive(Stage)]` codegen and Metal emission, not direct Rust reads.
#[allow(dead_code)]
pub struct FfnSwigluWriteStage {
    #[arg(buffer = 5, output, metal_type = "device OutputStorageT*")]
    output: TensorArg,
    #[arg(buffer = 11, metal_type = "const device BiasStorageT*")]
    b_gate: TensorArg,
    #[arg(buffer = 12, metal_type = "const device BiasStorageT*")]
    b_up: TensorArg,
    #[arg(buffer = 13)]
    has_b_gate: u32,
    #[arg(buffer = 14)]
    has_b_up: u32,
    #[arg(stage_skip)]
    activation: Activation,
}

impl Default for FfnSwigluWriteStage {
    fn default() -> Self {
        Self::new()
    }
}

impl FfnSwigluWriteStage {
    pub fn new() -> Self {
        Self {
            output: TensorArg::default(),
            b_gate: TensorArg::default(),
            b_up: TensorArg::default(),
            has_b_gate: 0,
            has_b_up: 0,
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

#[path = "ffn_stages.test.rs"]
mod tests;
