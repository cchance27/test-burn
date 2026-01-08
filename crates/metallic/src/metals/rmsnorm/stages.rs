use crate::compound::{BufferArg, Stage, stages::Quantization};

pub struct RmsNormComputeStage {
    pub input_buffer: usize,
    pub k_dim_buffer: usize,
    pub quantization: Quantization,
}

impl RmsNormComputeStage {
    pub fn new(input_buffer: usize, k_dim_buffer: usize) -> Self {
        Self {
            input_buffer,
            k_dim_buffer,
            quantization: Quantization::F16, // Default - overridden by with_quantization()
        }
    }

    pub fn with_quantization(mut self, q: Quantization) -> Self {
        self.quantization = q;
        self
    }
}

impl Stage for RmsNormComputeStage {
    fn includes(&self) -> Vec<&'static str> {
        // Including policy via Stage includes now
        vec![self.quantization.include_path()]
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
        ]
    }

    fn struct_defs(&self) -> String {
        // RmsNormParams struct + rmsnorm.metal functions
        // Policy is provided via includes()
        format!("{}\n{}", super::RmsNormParams::METAL_STRUCT_DEF, include_str!("rmsnorm.metal"))
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let policy = self.quantization.policy_name();
        // Cast half* input to uchar* for Policy interface compatibility
        let code = format!(
            r#"
    // --- RmsNorm Compute Stage ---
    // Compute inv_rms of the input vector for Apply fusion
    threadgroup float tg_inv_rms_storage;
    float inv_rms = rmsnorm_compute_inv_rms<{policy}>(
        (const device uchar*)input, 
        (const device uchar*)input, // Dummy scale_bytes (F16 ignores)
        k_dim, // feature_dim
        0,     // row_idx (Input is vector, always row 0)
        lane_id,
        warp_id,
        &tg_inv_rms_storage
    );
        "#,
            policy = policy
        );

        ("inv_rms".to_string(), code)
    }
}
