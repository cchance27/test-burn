use metallic_macros::{KernelArgs, Stage as DeriveStage};

use crate::{metals::rope::RopeParams, types::TensorArg};

/// Applies Rotary Position Embedding (RoPE) to Q as a pipeline stage.
#[derive(KernelArgs, Clone, Debug, DeriveStage)]
#[stage(
    includes("rope/rope_common.metal"),
    struct_defs = "RopeParams",
    emit = r#"
    threadgroup half4 q_shared[ROPE_MAX_HEAD_DIM_VEC];
    run_rope_decode_stage(q_shared, q_ptr, cos, sin, params_rope, lid.x);
    const threadgroup half4* q_ptr_roped = q_shared;
"#,
    out_var = "q_ptr_roped"
)]
pub struct RopeStage {
    #[arg(buffer = 12, metal_type = "const device half*")]
    pub cos: TensorArg,
    #[arg(buffer = 13, metal_type = "const device half*")]
    pub sin: TensorArg,
    #[arg(buffer = 14, metal_type = "constant RopeParams&")]
    pub params_rope: RopeParams,
}

impl RopeStage {
    pub fn new(cos: TensorArg, sin: TensorArg, params_rope: RopeParams) -> Self {
        Self { cos, sin, params_rope }
    }
}
