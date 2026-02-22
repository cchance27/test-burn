//! Arange Kernel for Foundry.
//!
//! Creates a tensor with sequential values: [0, 1, 2, ..., n-1]

use metallic_macros::KernelArgs;

use crate::{
    Includes, Kernel, KernelSource, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Arange kernel.
///
/// Fills output with sequential values starting from 0.
#[derive(KernelArgs, Clone)]
pub struct Arange {
    /// Output tensor.
    #[arg(output)]
    pub output: TensorArg,
    /// Length of the output.
    pub length: usize,
}

impl Arange {
    /// Create a new arange kernel.
    pub fn new(output: &TensorArg, length: usize) -> Self {
        Self {
            output: output.clone(),
            length,
        }
    }
}

impl Kernel for Arange {
    type Args = ();

    fn source(&self) -> KernelSource {
        KernelSource::File("tensor/arange.metal")
    }

    fn function_name(&self) -> &str {
        "arange_kernel_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)
    }

    fn struct_defs(&self) -> String {
        String::new() // No params struct
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
