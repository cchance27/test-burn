//! RMSNorm Kernel - Root Mean Square Layer Normalization.
//!
//! RMSNorm computes: output = (input / rms(input)) * gamma
//! where rms(x) = sqrt(mean(x^2) + eps)

use metallic_macros::MetalStruct;

pub mod stages;
pub mod step;

use crate::spec::DynamicValue;

/// Parameters for RMSNorm kernel.
#[derive(Clone, Debug, MetalStruct, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct RmsNormParams {
    /// Feature dimension (last dimension of input).
    pub feature_dim: DynamicValue<u32>,
    /// Total number of elements in input tensor.
    pub total_elements: DynamicValue<u32>,
    /// Epsilon for numerical stability.
    pub epsilon: DynamicValue<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compound::CompoundKernel, metals::rmsnorm::step::RmsNormStandaloneStage};

    #[test]
    fn test_rmsnorm_params_metal_struct() {
        let def = RmsNormParams::METAL_STRUCT_DEF;
        assert!(def.contains("uint feature_dim;"), "Should have feature_dim: {}", def);
        assert!(def.contains("uint total_elements;"), "Should have total_elements: {}", def);
        assert!(def.contains("float epsilon;"), "Should have epsilon: {}", def);
        assert!(def.contains("struct RmsNormParams"), "Should have struct name: {}", def);
    }

    #[test]
    fn test_rmsnorm_stage_fused() {
        let stage = RmsNormStandaloneStage::default();
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
        assert!(source.contains("RMSNORM_RUN_CORE_STAGE("));
    }
}
