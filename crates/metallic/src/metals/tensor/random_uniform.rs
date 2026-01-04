//! Random Uniform Kernel for Foundry.
//!
//! Creates a tensor filled with random values in [min, max).

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Parameters for RandomUniform kernel.
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct RandomUniformParams {
    /// RNG seed.
    pub seed: u32,
    /// Minimum value.
    pub min_val: f32,
    /// Scale (max - min).
    pub scale: f32,
}

/// Random Uniform kernel.
///
/// Fills output with random values in [min, min + scale).
#[derive(KernelArgs, Clone)]
pub struct RandomUniform {
    /// Output tensor.
    #[arg(output)]
    pub output: TensorArg,
    /// Kernel parameters.
    pub params: RandomUniformParams,
    /// Number of elements.
    pub length: usize,
}

impl RandomUniform {
    /// Create a new random uniform kernel.
    pub fn new(output: &TensorArg, length: usize, seed: u32, min_val: f32, max_val: f32) -> Self {
        Self {
            output: output.clone(),
            params: RandomUniformParams {
                seed,
                min_val,
                scale: max_val - min_val,
            },
            length,
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct RandomUniformId;

impl Kernel for RandomUniform {
    type Args = RandomUniformParams;
    type Id = RandomUniformId;

    fn source(&self) -> KernelSource {
        KernelSource::File("tensor/random_uniform.metal")
    }

    fn function_name(&self) -> &'static str {
        "random_uniform_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        RandomUniformParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let threads_per_group = 256;
        let num_groups = (self.length + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("RandomUniform kernel does not yet support compound kernel staging")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_uniform_params_metal_struct() {
        let def = RandomUniformParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct RandomUniformParams"));
        assert!(def.contains("seed"));
        assert!(def.contains("min_val"));
    }
}
