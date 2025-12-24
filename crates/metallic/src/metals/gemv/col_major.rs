//! GEMV Column-Major Kernel.
//! K dimension is contiguous, allowing vector loads.

use metallic_macros::KernelArgs;

pub use super::GemvParams;
use crate::{
    compound::{GemvCoreStage, Stage}, foundry::{Includes, Kernel, KernelSource}, tensor::Dtype, types::{DispatchConfig, GridSize, KernelArg, TensorArg, ThreadgroupSize}
};

/// GEMV Column-Major kernel for column-major matrix storage.
#[derive(KernelArgs, Clone)]
pub struct GemvColMajor {
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

impl GemvColMajor {
    /// Create a new GEMV Column-Major kernel without bias/residual.
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

    /// Create a GEMV Column-Major kernel with bias.
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
pub struct GemvColMajorId;

impl Kernel for GemvColMajor {
    type Args = Self;
    type Id = GemvColMajorId;

    fn source(&self) -> KernelSource {
        KernelSource::File("gemv/col_major.metal")
    }

    fn function_name(&self) -> &'static str {
        "gemv_col_major_f16"
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
        const COLS_PER_TG: usize = 8;
        const TG_WIDTH: usize = 256;

        let n = self.params.n as usize;
        let batch = self.params.batch.max(1) as usize;
        let num_tgs = (n + COLS_PER_TG - 1) / COLS_PER_TG;

        DispatchConfig {
            grid: GridSize::new(num_tgs, 1, batch),
            group: ThreadgroupSize::d1(TG_WIDTH),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        Box::new(GemvCoreStage::new_col_major())
    }
}
