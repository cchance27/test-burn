//! Random Uniform Kernel for Foundry.
//!
//! Creates a tensor filled with random values in [min, max).

use metallic_macros::{KernelArgs, MetalStruct};

use crate::{
    Includes, Kernel, KernelSource, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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
    #[arg(output, metal_type = "device OutputStorageT*")]
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

impl Kernel for RandomUniform {
    type Args = RandomUniformParams;

    fn source(&self) -> KernelSource {
        KernelSource::File("tensor/random_uniform.metal")
    }

    fn function_name(&self) -> &str {
        "random_uniform_kernel"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(self.output.dtype())
    }

    fn struct_defs(&self) -> String {
        RandomUniformParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let threads_per_group = 256;
        let num_groups = self.length.div_ceil(threads_per_group);

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }
}

#[path = "random_uniform.test.rs"]
mod tests;
