//! SwiGLU Fused Activation Kernel for Foundry.
//!
//! Computes: output = SiLU(gate + gate_bias) * (up + up_bias)
//! where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! This is the fused activation kernel used in transformer MLP blocks.
//! Note: The full SwiGLU operation (including matmuls) is a composite operation.

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for SwiGLU fused activation kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct SwigluParams {
    /// Total output elements.
    pub total_elements: u32,
    /// Bias length (hidden_dim).
    pub bias_len: u32,
    /// Vector width (4 for vectorized, 1 for scalar).
    pub vector_width: u32,
    /// Gate tensor leading stride.
    pub gate_leading_stride: u32,
    /// Up tensor leading stride.
    pub up_leading_stride: u32,
}

/// SwiGLU fused activation kernel.
///
/// Computes in-place: up_inout = SiLU(gate + gate_bias) * (up_inout + up_bias)
#[derive(KernelArgs, Clone)]
pub struct SwigluFusedActivation {
    /// Gate projection output.
    #[arg(buffer = 0)]
    pub gate: TensorArg,
    /// Up projection output (modified in-place).
    #[arg(buffer = 1, output)]
    pub up_inout: TensorArg,
    /// Gate bias.
    #[arg(buffer = 2)]
    pub gate_bias: TensorArg,
    /// Up bias.
    #[arg(buffer = 3)]
    pub up_bias: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 4)]
    pub params: SwigluParams,
}

impl SwigluFusedActivation {
    /// Create a new SwiGLU fused activation kernel.
    pub fn new(gate: &TensorArg, up_inout: &TensorArg, gate_bias: &TensorArg, up_bias: &TensorArg, params: SwigluParams) -> Self {
        Self {
            gate: gate.clone(),
            up_inout: up_inout.clone(),
            gate_bias: gate_bias.clone(),
            up_bias: up_bias.clone(),
            params,
        }
    }

    /// Create with automatic vectorization detection.
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
        let vector_width = if bias_len % 4 == 0 { 4 } else { 1 };

        Self::new(
            gate,
            up_inout,
            gate_bias,
            up_bias,
            SwigluParams {
                total_elements,
                bias_len,
                vector_width,
                gate_leading_stride,
                up_leading_stride,
            },
        )
    }
}

/// Kernel ID for pipeline caching.
pub struct SwigluFusedActivationId;

impl Kernel for SwigluFusedActivation {
    type Args = SwigluParams;
    type Id = SwigluFusedActivationId;

    fn source(&self) -> KernelSource {
        KernelSource::File("swiglu/swiglu.metal")
    }

    fn function_name(&self) -> &'static str {
        "swiglu_fused_activation_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        SwigluParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
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

        let num_groups = (total_threads + threads_per_group_width - 1) / threads_per_group_width;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group_width),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("SwiGLU kernel does not yet support compound kernel staging - needs Metal template refactoring")
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
