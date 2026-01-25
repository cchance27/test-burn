use metallic_macros::{Kernel, KernelArgs};
// Ensure typetag is available for macro expansion
use typetag;

use crate::{metals::mma::stages::TileConfig, policy::activation::Activation, spec::DynamicValue, types::TensorArg};

pub mod step;
pub use step::GemmParams; // Re-export Params
pub use step::*; // Re-export Step (which will implement CompiledStep for generated CompiledGemmV2Step)

#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "gemm/gemm_v2.metal", // Placeholder/Dynamically generated
    function = "gemm_v2", // Placeholder
    args = Self,
    step = true,
    execute = false
)]
pub struct GemmV2 {
    #[arg(buffer = 0)]
    pub a: TensorArg,
    #[arg(buffer = 1)]
    pub b: TensorArg,
    #[arg(buffer = 2, output)]
    pub d: TensorArg, // Output
    #[arg(buffer = 3)]
    pub c: Option<TensorArg>, // Residual
    #[arg(buffer = 4)]
    pub bias: Option<TensorArg>,
    #[arg(buffer = 5)]
    pub b_scales: Option<TensorArg>, // Explicit scales

    #[arg(meta, scale_for = "b")]
    pub derived_b_scales: TensorArg, // Implicit fallback scales

    #[arg(buffer = 6)]
    pub weights_per_block: u32,
    #[arg(buffer = 7)]
    pub alpha: f32,
    #[arg(buffer = 8)]
    pub beta: f32,
    #[arg(buffer = 10, serde_default)]
    pub params: GemmParams,

    // Meta fields for configuration
    #[arg(meta)]
    pub m_dim: DynamicValue<u32>,
    #[arg(meta)]
    pub n_dim: DynamicValue<u32>,
    #[arg(meta)]
    pub k_dim: DynamicValue<u32>,

    #[arg(meta, serde_default)]
    pub transpose_a: bool,
    #[arg(meta, serde_default)]
    pub transpose_b: bool,

    #[arg(meta, serde_default)]
    pub tile_config: Option<TileConfig>,

    #[arg(meta, serde_default)]
    pub activation: Activation,
}

impl Default for GemmV2 {
    fn default() -> Self {
        Self {
            a: Default::default(),
            b: Default::default(),
            d: Default::default(),
            c: None,
            bias: None,
            b_scales: None,
            derived_b_scales: Default::default(),
            weights_per_block: 32,
            alpha: 1.0,
            beta: 0.0,
            params: Default::default(),
            m_dim: DynamicValue::Literal(1),
            n_dim: DynamicValue::Literal(1),
            k_dim: DynamicValue::Literal(1),
            transpose_a: false,
            transpose_b: false,
            tile_config: None,
            activation: Activation::None,
        }
    }
}

impl GemmV2 {
    // Forward struct definition from GemmParams so the kernel has access to it
    pub const METAL_STRUCT_DEF: &'static str = GemmParams::METAL_STRUCT_DEF;
}
