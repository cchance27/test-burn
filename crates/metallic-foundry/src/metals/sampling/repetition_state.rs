use metallic_macros::{Kernel, KernelArgs, MetalStruct};

use crate::types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize};

#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct RepetitionStateParams {
    pub token_len: u32,
    pub window_len: u32,
}

#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "sampling/repetition_state.metal",
    function = "repetition_state_init_u32",
    args = RepetitionStateParams,
    dispatch = true,
    dtype = U32
)]
pub struct RepetitionStateInit {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub ring: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub pairs: TensorArg,
    #[arg(buffer = 2, output, metal_type = "device uint*")]
    pub meta: TensorArg,
    #[arg(buffer = 3, metal_type = "const device uint*")]
    pub tokens: TensorArg,
    pub params: RepetitionStateParams,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl RepetitionStateInit {
    pub fn new(ring: &TensorArg, pairs: &TensorArg, meta: &TensorArg, tokens: &TensorArg, token_len: u32, window_len: u32) -> Self {
        Self {
            ring: ring.clone(),
            pairs: pairs.clone(),
            meta: meta.clone(),
            tokens: tokens.clone(),
            params: RepetitionStateParams { token_len, window_len },
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
    source = "sampling/repetition_state.metal",
    function = "repetition_state_ingest_u32",
    args = RepetitionStateParams,
    dispatch = true,
    dtype = U32
)]
pub struct RepetitionStateIngest {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub ring: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub pairs: TensorArg,
    #[arg(buffer = 2, output, metal_type = "device uint*")]
    pub meta: TensorArg,
    #[arg(buffer = 3, metal_type = "const device uint*")]
    pub tokens: TensorArg,
    pub params: RepetitionStateParams,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl RepetitionStateIngest {
    pub fn new(ring: &TensorArg, pairs: &TensorArg, meta: &TensorArg, tokens: &TensorArg, token_len: u32, window_len: u32) -> Self {
        Self {
            ring: ring.clone(),
            pairs: pairs.clone(),
            meta: meta.clone(),
            tokens: tokens.clone(),
            params: RepetitionStateParams { token_len, window_len },
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
    source = "sampling/repetition_state.metal",
    function = "repetition_state_update_from_token_u32",
    args = RepetitionStateParams,
    dispatch = true,
    dtype = U32
)]
pub struct RepetitionStateUpdateFromToken {
    #[arg(buffer = 0, output, metal_type = "device uint*")]
    pub ring: TensorArg,
    #[arg(buffer = 1, output, metal_type = "device uint*")]
    pub pairs: TensorArg,
    #[arg(buffer = 2, output, metal_type = "device uint*")]
    pub meta: TensorArg,
    #[arg(buffer = 3, metal_type = "const device uint*")]
    pub token_buf: TensorArg,
    pub params: RepetitionStateParams,
    #[arg(skip)]
    pub threads_per_threadgroup: usize,
}

impl RepetitionStateUpdateFromToken {
    pub fn new(ring: &TensorArg, pairs: &TensorArg, meta: &TensorArg, token_buf: &TensorArg, window_len: u32) -> Self {
        Self {
            ring: ring.clone(),
            pairs: pairs.clone(),
            meta: meta.clone(),
            token_buf: token_buf.clone(),
            params: RepetitionStateParams { token_len: 1, window_len },
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
