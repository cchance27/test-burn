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

#[path = "mod.test.rs"]
mod tests;
