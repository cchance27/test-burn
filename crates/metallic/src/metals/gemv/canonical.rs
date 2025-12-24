//! GEMV Canonical Kernel (k-block-major layout).
//! Optimized for inference with pre-arranged weights.

use metallic_macros::KernelArgs;

pub use super::GemvParams;
use crate::{
    compound::{GemvCoreStage, Stage}, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{DispatchConfig, GridSize, KernelArg, TensorArg, ThreadgroupSize}
};

/// GEMV Canonical kernel for k-block-major weight storage.
#[derive(KernelArgs, Clone)]
pub struct GemvCanonical {
    #[arg(buffer = 0)]
    pub matrix: TensorArg,
    #[arg(buffer = 1)]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 2)]
    pub vector_x: TensorArg,
    #[arg(buffer = 3, output)]
    pub result_y: TensorArg,
    #[arg(buffer = 4)]
    pub params: GemvParams,
    #[arg(buffer = 5)]
    pub bias: TensorArg,
    #[arg(buffer = 6)]
    pub residual: TensorArg,
    #[arg(buffer = 7)]
    pub alpha: f32,
    #[arg(buffer = 8)]
    pub beta: f32,
    #[arg(buffer = 9)]
    pub has_bias: u32,
}

impl GemvCanonical {
    /// Create a new GEMV Canonical kernel without bias/residual.
    ///
    /// Note: `params.weights_per_block` must be set correctly for the canonical format.
    /// Typical values are 32 or 64.
    pub fn new(matrix: &TensorArg, vector_x: &TensorArg, result_y: &TensorArg, params: GemvParams) -> Self {
        Self {
            matrix: matrix.clone(),
            scale_bytes: matrix.clone(), // Dummy
            vector_x: vector_x.clone(),
            result_y: result_y.clone(),
            params,
            bias: matrix.clone(),
            residual: matrix.clone(),
            alpha: 1.0,
            beta: 0.0,
            has_bias: 0,
        }
    }

    /// Create a GEMV Canonical kernel with bias.
    pub fn with_bias(matrix: &TensorArg, vector_x: &TensorArg, result_y: &TensorArg, params: GemvParams, bias: &TensorArg) -> Self {
        Self {
            matrix: matrix.clone(),
            scale_bytes: matrix.clone(), // Dummy
            vector_x: vector_x.clone(),
            result_y: result_y.clone(),
            params,
            bias: bias.clone(),
            residual: matrix.clone(),
            alpha: 1.0,
            beta: 0.0,
            has_bias: 1,
        }
    }

    /// Set scale bytes for quantized weights.
    pub fn with_scales(mut self, scale_bytes: &TensorArg) -> Self {
        self.scale_bytes = scale_bytes.clone();
        self
    }

    /// Set residual tensor for residual connection.
    pub fn with_residual(mut self, residual: &TensorArg, beta: f32) -> Self {
        self.residual = residual.clone();
        self.beta = beta;
        self
    }

    /// Set alpha scaling factor.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Kernel ID for pipeline caching.
pub struct GemvCanonicalId;

impl Kernel for GemvCanonical {
    type Args = Self;
    type Id = GemvCanonicalId;

    fn source(&self) -> KernelSource {
        KernelSource::File("gemv/canonical.metal")
    }

    fn function_name(&self) -> &'static str {
        "gemv_canonical_f16"
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn dtype(&self) -> Option<Dtype> {
        Some(self.matrix.dtype())
    }

    fn struct_defs(&self) -> String {
        GemvParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &crate::types::ComputeCommandEncoder) {
        self.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        // Canonical uses 4 cols per TG (128 threads) - different from Dense/RowMajor
        const COLS_PER_TG: usize = 4;
        const TG_WIDTH: usize = 128; // 4 warps * 32 threads

        let n = self.params.n as usize;
        let batch = self.params.batch.max(1) as usize;
        let num_tgs = (n + COLS_PER_TG - 1) / COLS_PER_TG;

        DispatchConfig {
            grid: GridSize::new(num_tgs, 1, batch),
            group: ThreadgroupSize::d1(TG_WIDTH),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        Box::new(GemvCoreStage::new_canonical())
    }
}
