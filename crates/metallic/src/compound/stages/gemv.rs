//! GemvCoreStage - Main GEMV computation stage.
//!
//! This stage generates the dot product computation, expecting dequanted
//! float4 weights from the policy stage.

use crate::compound::{BufferArg, Stage};

/// The core GEMV computation stage.
///
/// Expects input_var = "w_vec" (float4 from policy's load_and_dequant).
/// Outputs "acc" (partial accumulator for this iteration).
#[derive(Clone)]
pub struct GemvCoreStage {
    /// Name of the core helper function to call (e.g., "run_gemv_row_major_core").
    pub core_function: &'static str,
    /// Buffer index for vector input.
    pub vector_buffer: u32,
    /// Buffer index for output.
    pub output_buffer: u32,
    /// Buffer index for params.
    pub params_buffer: u32,
    /// Metadata for policy header/defines (if any)
    pub includes: Vec<&'static str>,
}

impl GemvCoreStage {
    pub fn new_row_major() -> Self {
        Self {
            core_function: "run_gemv_row_major_core",
            vector_buffer: 2,
            output_buffer: 3,
            params_buffer: 4,
            includes: vec!["gemv/row_major.metal"],
        }
    }

    pub fn new_col_major() -> Self {
        Self {
            core_function: "run_gemv_col_major_core",
            vector_buffer: 2,
            output_buffer: 3,
            params_buffer: 4,
            includes: vec!["gemv/col_major.metal"],
        }
    }

    pub fn new_canonical() -> Self {
        Self {
            core_function: "run_gemv_canonical_core",
            vector_buffer: 2,
            output_buffer: 3,
            params_buffer: 4,
            includes: vec!["gemv/canonical.metal"],
        }
    }
}

impl Stage for GemvCoreStage {
    fn includes(&self) -> Vec<&'static str> {
        self.includes.clone()
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "vector_x",
                metal_type: "const device half*",
                buffer_index: self.vector_buffer,
            },
            BufferArg {
                name: "result_y",
                metal_type: "device half*",
                buffer_index: self.output_buffer,
            },
            BufferArg {
                name: "params",
                metal_type: "const constant GemvParams*",
                buffer_index: self.params_buffer,
            },
            BufferArg {
                name: "bias",
                metal_type: "const device half*",
                buffer_index: self.params_buffer + 1, // 5
            },
            BufferArg {
                name: "residual",
                metal_type: "const device half*",
                buffer_index: self.params_buffer + 2, // 6
            },
            BufferArg {
                name: "alpha",
                metal_type: "constant float&",
                buffer_index: self.params_buffer + 3, // 7
            },
            BufferArg {
                name: "beta",
                metal_type: "constant float&",
                buffer_index: self.params_buffer + 4, // 8
            },
            BufferArg {
                name: "has_bias",
                metal_type: "constant uint&",
                buffer_index: self.params_buffer + 5, // 9
            },
        ]
    }

    fn struct_defs(&self) -> String {
        use crate::metals::gemv::GemvParams;
        GemvParams::METAL_STRUCT_DEF.to_string()
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let core = self.core_function;
        let code = format!(
            r#"
    if (has_bias != 0) {{
        {core}<Policy, true>(matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes);
    }} else {{
        {core}<Policy, false>(matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes);
    }}"#
        );

        ("void".to_string(), code)
    }
}
