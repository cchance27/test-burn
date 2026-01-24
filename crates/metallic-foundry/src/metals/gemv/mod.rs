use metallic_macros::{Kernel, KernelArgs, MetalStruct};

pub mod fused_step;
pub mod qkv_stages;
pub mod qkv_step;
pub mod stages;
pub mod step;

pub use fused_step::*;
pub use step::*;

use crate::{compound::Layout, policy::activation::Activation, spec::DynamicValue, types::TensorArg};

#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct GemvV2Params {
    pub k_dim: DynamicValue<u32>,
    pub n_dim: DynamicValue<u32>,
    #[serde(default = "default_wpb")]
    pub weights_per_block: u32,
    #[serde(default = "default_batch")]
    pub batch: u32,
}

fn default_wpb() -> u32 {
    32
}
fn default_batch() -> u32 {
    1
}

#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "gemv/gemv.metal", // Placeholder
    function = "gemv_kernel",   // Placeholder
    args = GemvV2Params,
    step = true,
    execute = false
)]
pub struct GemvV2 {
    #[arg(buffer = 0)]
    pub weights: TensorArg,
    #[arg(buffer = 1)]
    pub scale_bytes: Option<TensorArg>, // Explicit scales via JSON (Q8 legacy/custom)

    #[arg(meta, scale_for = "weights")]
    pub derived_scales: TensorArg, // Implicitly derived scales (Q8 fallback)
    #[arg(buffer = 2)]
    pub input: TensorArg,
    #[arg(buffer = 3, output)]
    pub output: TensorArg,

    // Meta fields for selection logic
    #[arg(meta)]
    pub layout: Layout,
    #[arg(meta)]
    pub strategy: Option<step::GemvStrategy>,
    #[arg(meta)]
    pub activation: Activation,

    // Explicit scalar args (in case we need them bound, though params covers most dynamic ones)
    // GemvV2Step had static alpha/beta.
    pub alpha: f32,
    pub beta: f32,

    pub has_bias: u32,
    pub has_residual: u32,
    pub bias: Option<TensorArg>,
    pub residual: Option<TensorArg>,

    // Dynamic params
    pub params: GemvV2ParamsResolved,
}
