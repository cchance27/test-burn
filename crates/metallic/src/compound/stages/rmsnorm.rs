//! RmsNormCoreStage - Main RMSNorm computation stage for compound kernels.
//!
//! This stage generates the RMSNorm normalization, using the policy for input loading.

use crate::compound::{BufferArg, Stage};

/// The core RMSNorm computation stage.
///
/// Uses Policy::load_weights/load_scale for input loading, supporting both F16 and Q8.
/// Buffer layout matches PolicyStage convention: matrix(0), scale_bytes(1).
#[derive(Clone)]
pub struct RmsNormCoreStage {
    /// Buffer index for matrix/input data (as uchar* for policy compat).
    pub matrix_buffer: u32,
    /// Buffer index for scale bytes (from PolicyStage).
    pub scale_buffer: u32,
    /// Buffer index for output.
    pub output_buffer: u32,
    /// Buffer index for gamma weights.
    pub gamma_buffer: u32,
    /// Buffer index for params.
    pub params_buffer: u32,
}

impl RmsNormCoreStage {
    /// Create with default buffer layout matching PolicyStage convention.
    /// matrix(0), scale_bytes(1), then RMSNorm-specific: output(2), gamma(3), params(4)
    pub fn new() -> Self {
        Self {
            matrix_buffer: 0,
            scale_buffer: 1,
            output_buffer: 2,
            gamma_buffer: 3,
            params_buffer: 4,
        }
    }

    /// Create with custom buffer indices.
    pub fn with_buffers(matrix: u32, scale: u32, output: u32, gamma: u32, params: u32) -> Self {
        Self {
            matrix_buffer: matrix,
            scale_buffer: scale,
            output_buffer: output,
            gamma_buffer: gamma,
            params_buffer: params,
        }
    }
}

impl Default for RmsNormCoreStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for RmsNormCoreStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rmsnorm/rmsnorm.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Note: matrix(0) and scale_bytes(1) come from PolicyStage
        // Only declare RMSNorm-specific buffers
        vec![
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: self.output_buffer,
            },
            BufferArg {
                name: "gamma",
                metal_type: "const device half*",
                buffer_index: self.gamma_buffer,
            },
            BufferArg {
                name: "params",
                metal_type: "const constant RmsNormParams*",
                buffer_index: self.params_buffer,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        use crate::metals::rmsnorm::RmsNormParams;
        RmsNormParams::METAL_STRUCT_DEF.to_string()
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        // Declare threadgroup for parallel reduction
        // Use 'matrix' (input) and 'scale_bytes' from policy convention
        let code = r#"
    threadgroup float tg_inv_rms;
    run_rmsnorm_core<Policy>(matrix, output, gamma, params, scale_bytes, gid, lid, &tg_inv_rms);"#
            .to_string();

        ("void".to_string(), code)
    }
}
