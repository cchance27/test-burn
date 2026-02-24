//! Ones Kernel for Foundry.
//!
//! Creates a tensor filled with 1.0.

use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::TensorArg;

/// Parameters for Ones kernel.
#[derive(MetalStruct, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct OnesParams {
    /// Total elements to fill.
    pub total_elements: u32,
}

/// Ones kernel.
///
/// Fills output with 1.0. Vectorized: each thread handles 4 elements.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "tensor/ones.metal",
    function = "ones_kernel",
    args = OnesParams,
    dispatch = vec_4,
    include_exprs("crate::policy::resolve_policy(self.output.dtype()).header()")
)]
pub struct Ones {
    /// Output tensor.
    #[arg(output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    /// Kernel parameters.
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

#[path = "ones.test.rs"]
mod tests;
