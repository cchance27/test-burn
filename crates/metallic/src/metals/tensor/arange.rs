//! Arange Kernel for Foundry.
//!
//! Creates a tensor with sequential values: [0, 1, 2, ..., n-1]

use metallic_macros::KernelArgs;

use crate::{
    compound::Stage, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// Arange kernel.
///
/// Fills output with sequential values starting from 0.
#[derive(KernelArgs, Clone)]
pub struct Arange {
    /// Output tensor.
    #[arg(buffer = 0, output)]
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

/// Kernel ID for pipeline caching.
pub struct ArangeId;

impl Kernel for Arange {
    type Args = ();
    type Id = ArangeId;

    fn source(&self) -> KernelSource {
        KernelSource::File("tensor/arange.metal")
    }

    fn function_name(&self) -> &'static str {
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
        let num_groups = (self.length + threads_per_group - 1) / threads_per_group;

        DispatchConfig {
            grid: GridSize::d1(num_groups),
            group: ThreadgroupSize::d1(threads_per_group),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        todo!("Arange kernel does not yet support compound kernel staging")
    }
}
