//! SwiGLU Fused Activation Kernel for Foundry.
//!
//! Computes: output = SiLU(gate + gate_bias) * (up + up_bias)
//! where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! This module provides:
//! - `Swiglu` - Standalone activation kernel
//! - `SwiGluEpilogue` - SIMD GEMV epilogue for fused decode path

use metallic_macros::{Epilogue, Kernel, KernelArgs, MetalStruct};

use crate::{
    spec::DynamicValue, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

pub mod ffn_stages;
pub mod stages;
pub mod step;

// ================================================================================================
// SIMD GEMV Epilogue (for fused decode path)
// ================================================================================================

/// SIMD GEMV template epilogue policy: fuses Gate+Up + SiLU activation.
///
/// This is *not* a scalar post-stage epilogue (it operates on the GEMV template's
/// per-warp accumulators), so we only implement `GemvEpilogue`.
#[derive(Epilogue, Clone, Copy, Default)]
#[epilogue(
    include = "swiglu/swiglu.metal",
    gemv_struct = "SwiGluEpilogue",
    gemv_id = "swiglu",
    simd_reduce = "gate: acc[0], up: acc[1]"
)]
pub struct SwiGluEpilogue;

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
#[kernel(source = "swiglu/swiglu.metal", function = "swiglu_fused_activation_f16", args = "SwigluParamsResolved")]
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

    pub fn dispatch_config(&self) -> DispatchConfig {
        let vector_width = std::cmp::max(self.params.vector_width as usize, 1);
        let base_threads = 256;
        let threads_per_group_width = std::cmp::max(base_threads / vector_width, 1);
        let total_threads = if self.params.vector_width > 1 {
            let vectorized = self.params.total_elements / self.params.vector_width;
            let remainder = self.params.total_elements % self.params.vector_width;
            (vectorized + remainder) as usize
        } else {
            self.params.total_elements as usize
        };
        let num_groups = total_threads.div_ceil(threads_per_group_width);
        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group_width),
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct SwigluId;

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
