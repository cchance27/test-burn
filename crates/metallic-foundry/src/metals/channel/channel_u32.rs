use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize};

#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct ChannelU32Params {
    pub capacity: u32,
}

#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "channel/channel_u32.metal",
    function = "channel_u32_init",
    args = ChannelU32Params,
    dispatch = true,
    dtype = U32
)]
pub struct ChannelU32Init {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub header: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub data: TensorArg,
    pub params: ChannelU32Params,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl ChannelU32Init {
    pub fn new(header: &TensorArg, data: &TensorArg, capacity: u32) -> Self {
        Self {
            header: header.clone(),
            data: data.clone(),
            params: ChannelU32Params { capacity },
            threads_per_threadgroup: 1,
        }
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            grid: GridSize::d1(1),
            group: ThreadgroupSize::d1(self.threads_per_threadgroup),
        }
    }
}

#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "channel/channel_u32.metal",
    function = "channel_u32_push",
    args = ChannelU32Params,
    dispatch = true,
    dtype = U32
)]
pub struct ChannelU32Push {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub header: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub data: TensorArg,
    #[arg(buffer = 2, metal_type = "const device uint*")]
    pub value: TensorArg,
    pub params: ChannelU32Params,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl ChannelU32Push {
    pub fn new(header: &TensorArg, data: &TensorArg, value: &TensorArg) -> Self {
        Self {
            header: header.clone(),
            data: data.clone(),
            value: value.clone(),
            params: ChannelU32Params { capacity: 0 },
            threads_per_threadgroup: 1,
        }
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            grid: GridSize::d1(1),
            group: ThreadgroupSize::d1(self.threads_per_threadgroup),
        }
    }
}

#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "channel/channel_u32.metal",
    function = "channel_u32_push_scalar",
    args = ChannelU32Params,
    dispatch = true,
    dtype = U32
)]
pub struct ChannelU32PushScalar {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub header: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub data: TensorArg,
    #[arg(buffer = 2)]
    pub value: u32,
    pub params: ChannelU32Params,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl ChannelU32PushScalar {
    pub fn new(header: &TensorArg, data: &TensorArg, value: u32) -> Self {
        Self {
            header: header.clone(),
            data: data.clone(),
            value,
            params: ChannelU32Params { capacity: 0 },
            threads_per_threadgroup: 1,
        }
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            grid: GridSize::d1(1),
            group: ThreadgroupSize::d1(self.threads_per_threadgroup),
        }
    }
}
