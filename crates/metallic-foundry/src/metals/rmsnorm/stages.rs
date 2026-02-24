use std::sync::Arc;

use metallic_macros::Stage;

use super::RmsNormParams;
use crate::{compound::BufferArg, fusion::MetalPolicy, policy::f16::PolicyF16};

#[derive(Debug, Clone, Stage)]
#[stage(
    includes("dtypes/runtime_types.metal", "rmsnorm/rmsnorm.metal"),
    struct_defs = "RmsNormParams",
    policy_field = "policy",
    buffer_args_fn = "stage_buffer_args",
    template_bindings(policy_struct = "self.policy.struct_name()"),
    emit = r#"
    RMSNORM_COMPUTE_INV_RMS_STAGE(
        {out_var},
        {policy_struct},
        (const device uchar*)input,
        (const device uchar*)input,
        k_dim,
        batch_idx,
        lane_id,
        warp_id,
        lid.x,
        tptg.x,
        epsilon
    );
"#,
    out_var = "inv_rms"
)]
pub struct RmsNormComputeStage {
    pub input_buffer: usize,
    pub k_dim_buffer: usize,
    pub epsilon_buffer: usize,
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
}

impl RmsNormComputeStage {
    /// Create an RMSNorm compute stage.
    ///
    /// Note: the arguments are Metal buffer indices in the generated kernel signature.
    pub fn new(input_buffer_index: usize, k_dim_buffer_index: usize, epsilon_buffer_index: usize) -> Self {
        Self {
            input_buffer: input_buffer_index,
            k_dim_buffer: k_dim_buffer_index,
            epsilon_buffer: epsilon_buffer_index,
            policy: Arc::new(PolicyF16),
        }
    }

    pub fn with_policy(mut self, policy: Arc<dyn MetalPolicy>) -> Self {
        self.policy = policy;
        self
    }

    fn stage_buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "input",
                metal_type: "const device InputStorageT*",
                buffer_index: self.input_buffer as u32,
            },
            BufferArg {
                name: "k_dim",
                metal_type: "constant uint&",
                buffer_index: self.k_dim_buffer as u32,
            },
            BufferArg {
                name: "epsilon",
                metal_type: "constant float&",
                buffer_index: self.epsilon_buffer as u32,
            },
        ]
    }
}
