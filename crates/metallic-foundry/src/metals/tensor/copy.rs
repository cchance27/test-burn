use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::TensorArg;

#[derive(MetalStruct, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct CopyParams {
    pub total_elements: u32,
}

#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "tensor/copy.metal",
    function = "copy_u32",
    args = CopyParams,
    dispatch = per_element,
    dtype = U32
)]
pub struct CopyU32 {
    #[arg(buffer = 0, metal_type = "const device uint*")]
    pub src: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub dst: TensorArg,
    pub params: CopyParams,
}

impl CopyU32 {
    pub fn new(src: TensorArg, dst: TensorArg, count: u32) -> Self {
        Self {
            src,
            dst,
            params: CopyParams { total_elements: count },
        }
    }
}
