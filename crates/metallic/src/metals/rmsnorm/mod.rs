//! RMSNorm Kernel - Root Mean Square Layer Normalization.
//!
//! RMSNorm computes: output = (input / rms(input)) * gamma
//! where rms(x) = sqrt(mean(x^2) + eps)

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for RMSNorm kernel.
#[derive(Clone, Copy, Debug, MetalStruct)]
#[repr(C)]
pub struct RmsNormParams {
    /// Feature dimension (last dimension of input).
    pub feature_dim: u32,
    /// Total number of elements in input tensor.
    pub total_elements: u32,
}

/// RMSNorm kernel.
///
/// Performs: output[i] = (input[i] / rms) * gamma[feature_idx]
/// where rms = sqrt(sum(input[row]^2) / feature_dim + eps)
#[derive(KernelArgs, Clone)]
pub struct RmsNorm {
    #[arg(buffer = 0)]
    pub input: TensorArg,
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    #[arg(buffer = 2)]
    pub gamma: TensorArg,
    #[arg(buffer = 3)]
    pub params: RmsNormParams,
}

impl RmsNorm {
    /// Create a new RMSNorm kernel.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [..., feature_dim]
    /// * `output` - Output tensor (same shape as input)
    /// * `gamma` - Scale weights of shape [feature_dim]
    /// * `params` - RMSNorm parameters (feature_dim, total_elements)
    pub fn new(input: &TensorArg, output: &TensorArg, gamma: &TensorArg, params: RmsNormParams) -> Self {
        Self {
            input: input.clone(),
            output: output.clone(),
            gamma: gamma.clone(),
            params,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct RmsNormId;

impl Kernel for RmsNorm {
    type Args = RmsNormParams;
    type Id = RmsNormId;

    fn source(&self) -> KernelSource {
        KernelSource::File("rmsnorm/rmsnorm.metal")
    }

    fn function_name(&self) -> &'static str {
        "rmsnorm_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        RmsNormParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &crate::types::ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // One threadgroup per row, 256 threads per threadgroup
        let num_rows = self.params.total_elements / self.params.feature_dim;
        let threads_per_group = 256;

        DispatchConfig {
            grid: GridSize::d1(num_rows as usize),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        Box::new(crate::compound::RmsNormCoreStage::new())
    }
}

/// RMSNorm stage for compound kernel fusion.
///
/// Can be used as an epilogue stage to apply RMSNorm after another operation.
#[derive(Clone)]
pub struct RmsNormStage {
    /// Buffer index for gamma weights.
    pub gamma_buffer: u32,
}

impl RmsNormStage {
    pub fn new() -> Self {
        Self { gamma_buffer: 2 }
    }

    pub fn with_gamma_buffer(mut self, idx: u32) -> Self {
        self.gamma_buffer = idx;
        self
    }
}

impl Default for RmsNormStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for RmsNormStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rmsnorm/rmsnorm.metal"]
    }

    fn buffer_args(&self) -> Vec<crate::compound::BufferArg> {
        vec![crate::compound::BufferArg {
            name: "gamma",
            metal_type: "const device half*",
            buffer_index: self.gamma_buffer,
        }]
    }

    fn struct_defs(&self) -> String {
        String::new() // No additional struct defs needed for stage
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        // Assumes we're operating on a single value with feature_idx available
        let code = format!(
            r#"
    // RMSNorm epilogue: scale by gamma
    half gamma_val = gamma[feature_idx];
    half {input_var}_normed = {input_var} * gamma_val;"#
        );
        (format!("{}_normed", input_var), code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_params_metal_struct() {
        let def = RmsNormParams::METAL_STRUCT_DEF;
        assert!(def.contains("uint feature_dim;"), "Should have feature_dim: {}", def);
        assert!(def.contains("uint total_elements;"), "Should have total_elements: {}", def);
        assert!(def.contains("struct RmsNormParams"), "Should have struct name: {}", def);
    }
}
