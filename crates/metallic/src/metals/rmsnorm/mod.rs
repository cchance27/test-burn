//! RMSNorm Kernel - Root Mean Square Layer Normalization.
//!
//! RMSNorm computes: output = (input / rms(input)) * gamma
//! where rms(x) = sqrt(mean(x^2) + eps)

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

pub mod stages;
pub mod step;

use crate::{foundry::spec::DynamicValue, types::TensorArg};

/// Parameters for RMSNorm kernel.
#[derive(Clone, Debug, MetalStruct, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct RmsNormParams {
    /// Feature dimension (last dimension of input).
    pub feature_dim: DynamicValue<u32>,
    /// Total number of elements in input tensor.
    pub total_elements: DynamicValue<u32>,
}

/// RMSNorm kernel.
///
/// RMSNorm is typically used as a **prologue** stage before the main computation:
/// - Pre-attention: `RMSNorm → QKV projection`
/// - Pre-MLP: `RMSNorm → Linear`
///
/// The `stage_function` allows this kernel to be used in fused compound kernels.
/// Gamma scaling is applied internally in the Metal shader (`rmsnorm_apply`).
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "rmsnorm/rmsnorm.metal",
    function = "rmsnorm_kernel_f16",
    stage_function = "run_rmsnorm_core",
    args = RmsNormParams,
    dispatch = per_row,
    threadgroup = "float tg_inv_rms"
)]
pub struct RmsNorm {
    /// Input tensor (Buffer 0 - Policy Matrix).
    #[arg(stage_skip)]
    pub input: TensorArg,
    /// Scale bytes for Q8 policy (Buffer 1 - Policy Scales).
    #[arg(stage_skip)]
    pub scale_bytes: TensorArg,
    /// Output tensor (Buffer 2).
    #[arg(output)]
    pub output: TensorArg,
    /// Scale weights (Buffer 3).
    pub gamma: TensorArg,
    /// RMSNorm parameters (Buffer 4).
    #[arg(metal_type = "const constant RmsNormParams*")]
    pub params: RmsNormParamsResolved,
}

impl RmsNorm {
    /// Create a new RMSNorm kernel.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [..., feature_dim]
    /// * `output` - Output tensor (same shape as input)
    /// * `gamma` - Scale weights of shape [feature_dim]
    /// * `params` - RMSNorm parameters (feature_dim, total_elements)
    pub fn new(input: &TensorArg, output: &TensorArg, gamma: &TensorArg, params: RmsNormParamsResolved) -> Self {
        Self {
            input: input.clone(),
            scale_bytes: input.clone(), // Default to input for scales (F16 case)
            output: output.clone(),
            gamma: gamma.clone(),
            params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compound::CompoundKernel;

    #[test]
    fn test_rmsnorm_params_metal_struct() {
        let def = RmsNormParams::METAL_STRUCT_DEF;
        assert!(def.contains("uint feature_dim;"), "Should have feature_dim: {}", def);
        assert!(def.contains("uint total_elements;"), "Should have total_elements: {}", def);
        assert!(def.contains("struct RmsNormParams"), "Should have struct name: {}", def);
    }

    #[test]
    fn test_rmsnorm_stage_fused() {
        let stage = RmsNormStage::default();
        let kernel = CompoundKernel::new("test_fused")
            .main_dyn(Box::new(stage))
            .with_manual_output(true)
            .build();

        let source = kernel.source();

        // Verify includes
        assert!(source.contains("#include \"rmsnorm/rmsnorm.metal\""));

        // Verify buffer arguments collected
        let args = kernel.collect_buffer_args();
        let gamma_arg = args.iter().find(|a| a.name == "gamma").expect("Should have gamma arg");
        assert_eq!(gamma_arg.buffer_index, 3);
        assert_eq!(gamma_arg.metal_type, "const device half*");

        // Verify code emission
        assert!(source.contains("threadgroup float tg_inv_rms"));
        assert!(source.contains("run_rmsnorm_core<Policy>("));
    }
}
