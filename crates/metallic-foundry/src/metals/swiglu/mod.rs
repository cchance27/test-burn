//! SwiGLU Fused Activation Kernel for Foundry.
//!
//! Computes: output = SiLU(gate + gate_bias) * (up + up_bias)
//! where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! This module provides:
//! - `Swiglu` - Standalone activation kernel

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::{spec::DynamicValue, types::TensorArg};

pub mod ffn_stages;
pub mod stages;
pub mod step;

/// Arguments for SwiGLU fused activation kernel (Stage variant).
#[derive(Debug, KernelArgs)]
pub struct SwigluArgs {
    #[arg(buffer = 0)]
    pub gate: TensorArg,
    #[arg(buffer = 1, output)]
    pub up_inout: TensorArg,
    #[arg(buffer = 2)]
    pub gate_bias: TensorArg,
    #[arg(buffer = 3)]
    pub up_bias: TensorArg,
    #[arg(buffer = 4)]
    pub params: SwigluParamsResolved,
}

// ================================================================================================
// Standalone Kernel
// ================================================================================================

/// Parameters for SwiGLU fused activation kernel.
#[derive(MetalStruct, Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
#[repr(C)]
pub struct SwigluParams {
    /// Total output elements.
    pub total_elements: DynamicValue<u32>,
    /// Bias length (hidden_dim).
    pub bias_len: DynamicValue<u32>,
    /// Vector width (4 for vectorized, 1 for scalar).
    pub vector_width: u32,
    /// Gate tensor leading stride.
    pub gate_leading_stride: DynamicValue<u32>,
    /// Up tensor leading stride.
    pub up_leading_stride: DynamicValue<u32>,
}

/// SwiGLU fused activation kernel.
///
/// Computes in-place: up_inout = SiLU(gate + gate_bias) * (up_inout + up_bias)
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "swiglu/swiglu.metal",
    function = "swiglu_fused_activation_f16",
    args = "SwigluParamsResolved",
    dispatch = per_element_vec,
    step = true,
    execute = false
)]
pub struct Swiglu {
    /// Gate projection output.
    pub gate: TensorArg,
    /// Up projection output (modified in-place).
    #[arg(output)]
    pub up_inout: TensorArg,
    /// Gate bias.
    pub gate_bias: TensorArg,
    /// Up bias.
    pub up_bias: TensorArg,
    /// Kernel parameters.
    pub params: SwigluParamsResolved,
}

impl Swiglu {
    /// Create a new SwiGLU fused activation kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new(gate: &TensorArg, up_inout: &TensorArg, gate_bias: &TensorArg, up_bias: &TensorArg, params: SwigluParamsResolved) -> Self {
        Self {
            gate: gate.clone(),
            up_inout: up_inout.clone(),
            gate_bias: gate_bias.clone(),
            up_bias: up_bias.clone(),
            params,
        }
    }

    /// Create with automatic vectorization detection.
    #[allow(clippy::too_many_arguments)]
    pub fn new_auto_vectorized(
        gate: &TensorArg,
        up_inout: &TensorArg,
        gate_bias: &TensorArg,
        up_bias: &TensorArg,
        total_elements: u32,
        bias_len: u32,
        gate_leading_stride: u32,
        up_leading_stride: u32,
    ) -> Self {
        let vector_width = if bias_len.is_multiple_of(4) { 4 } else { 1 };

        Self::new(
            gate,
            up_inout,
            gate_bias,
            up_bias,
            SwigluParamsResolved {
                total_elements,
                bias_len,
                vector_width,
                gate_leading_stride,
                up_leading_stride,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_params_metal_struct() {
        let def = SwigluParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct SwigluParams"));
        assert!(def.contains("total_elements"));
        assert!(def.contains("bias_len"));
        assert!(def.contains("vector_width"));
    }
}
