use std::sync::Arc;

use crate::{
    compound::{BufferArg, Stage}, fusion::MetalPolicy
};

pub struct RmsNormComputeStage {
    pub input_buffer: usize,
    pub k_dim_buffer: usize,
    pub epsilon_buffer: usize,
    pub policy: Arc<dyn MetalPolicy>,
}

impl RmsNormComputeStage {
    /// Create an RMSNorm compute stage.
    ///
    /// Note: the arguments are **Metal buffer indices** in the generated kernel signature,
    /// not tensor dimensions.
    pub fn new(input_buffer_index: usize, k_dim_buffer_index: usize, epsilon_buffer_index: usize) -> Self {
        Self {
            input_buffer: input_buffer_index,
            k_dim_buffer: k_dim_buffer_index,
            epsilon_buffer: epsilon_buffer_index,
            policy: std::sync::Arc::new(crate::policy::f16::PolicyF16), // Default - overridden by with_policy()
        }
    }

    pub fn with_policy(mut self, policy: Arc<dyn MetalPolicy>) -> Self {
        self.policy = policy;
        self
    }
}

impl Stage for RmsNormComputeStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![self.policy.header()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "input",
                metal_type: "const device half*", // Activation input is half*
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

    fn struct_defs(&self) -> String {
        // RmsNormParams struct + rmsnorm.metal functions
        // Policy is provided via includes()
        format!("{}\n{}", super::RmsNormParams::METAL_STRUCT_DEF, include_str!("rmsnorm.metal"))
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.policy.struct_name();
        // Cast half* input to uchar* for Policy interface compatibility
        let code = format!(
            r#"
    // --- RmsNorm Compute Stage ---
    // Compute inv_rms of the input vector for Apply fusion
    threadgroup float tg_inv_rms_storage;
    if (lane_id == 0u && warp_id == 0u) {{
        tg_inv_rms_storage = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rmsnorm_compute_inv_rms<{policy}>(
        (const device uchar*)input, 
        (const device uchar*)input, // Dummy scale_bytes (F16 ignores)
        k_dim, // feature_dim
        batch_idx, // row_idx (batched: one row per token)
        lane_id,
        warp_id,
        epsilon,
        &tg_inv_rms_storage
    );
        "#,
            policy = policy
        );

        ("inv_rms".to_string(), code)
    }
}
