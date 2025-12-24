//! Broadcast Element-wise Add Kernel for Foundry.
//!
//! Adds a 1D bias tensor to each row: out[i] = a[i] + b[i % b_len]

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for ElemwiseAdd kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct ElemwiseAddParams {
    /// Total elements in output.
    pub total_elements: u32,
    /// Length of bias tensor (for broadcast).
    pub b_len: u32,
}

/// Broadcast element-wise add kernel.
///
/// out[i] = a[i] + b[i % b_len]
#[derive(KernelArgs, Clone)]
pub struct ElemwiseAdd {
    /// Input tensor a.
    #[arg(buffer = 0)]
    pub a: TensorArg,
    /// Bias tensor b (1D).
    #[arg(buffer = 1)]
    pub b: TensorArg,
    /// Output tensor.
    #[arg(buffer = 2, output)]
    pub out: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 3)]
    pub params: ElemwiseAddParams,
}

impl ElemwiseAdd {
    /// Create a new broadcast add kernel.
    pub fn new(a: &TensorArg, b: &TensorArg, out: &TensorArg, params: ElemwiseAddParams) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
            out: out.clone(),
            params,
        }
    }

    /// Create inplace variant (out = a).
    pub fn new_inplace(a: &TensorArg, b: &TensorArg, total_elements: u32, b_len: u32) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
            out: a.clone(), // Same as input for inplace
            params: ElemwiseAddParams { total_elements, b_len },
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct ElemwiseAddId;

impl Kernel for ElemwiseAdd {
    type Args = ElemwiseAddParams;
    type Id = ElemwiseAddId;

    fn source(&self) -> KernelSource {
        KernelSource::File("elemwise/add.metal")
    }

    fn function_name(&self) -> &'static str {
        "broadcast_add_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        ElemwiseAddParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let total = self.params.total_elements as usize;
        let threads_per_group = 256;
        let num_groups = (total + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("ElemwiseAdd kernel does not yet support compound kernel staging")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elemwise_add_params_metal_struct() {
        let def = ElemwiseAddParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct ElemwiseAddParams"));
        assert!(def.contains("total_elements"));
    }
}
