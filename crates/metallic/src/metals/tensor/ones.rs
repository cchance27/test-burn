//! Ones Kernel for Foundry.
//!
//! Creates a tensor filled with 1.0.

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for Ones kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct OnesParams {
    /// Total elements to fill.
    pub total_elements: u32,
}

/// Ones kernel.
///
/// Fills output with 1.0. Vectorized: each thread handles 4 elements.
#[derive(KernelArgs, Clone)]
pub struct Ones {
    /// Output tensor.
    #[arg(buffer = 0, output)]
    pub output: TensorArg,
    /// Kernel parameters.
    #[arg(buffer = 1)]
    pub params: OnesParams,
}

impl Ones {
    /// Create a new ones kernel.
    pub fn new(output: &TensorArg, total_elements: u32) -> Self {
        Self {
            output: output.clone(),
            params: OnesParams { total_elements },
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct OnesId;

impl Kernel for Ones {
    type Args = OnesParams;
    type Id = OnesId;

    fn source(&self) -> KernelSource {
        KernelSource::File("tensor/ones.metal")
    }

    fn function_name(&self) -> &'static str {
        "ones_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        OnesParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // Each thread handles 4 elements
        let total_threads = (self.params.total_elements as usize + 3) / 4;
        let threads_per_group = 256;
        let num_groups = (total_threads + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("Ones kernel does not yet support compound kernel staging")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ones_params_metal_struct() {
        let def = OnesParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct OnesParams"));
        assert!(def.contains("total_elements"));
    }
}
