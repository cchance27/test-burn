//! GEMV Canonical Kernel (k-block-major layout).
//! Optimized for inference with pre-arranged weights.

use metallic_macros::{Kernel, KernelArgs};

pub use super::GemvParams;
use crate::types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize};

/// GEMV Canonical kernel for k-block-major weight storage.
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "gemv/canonical.metal",
    function = "gemv_canonical_f16",
    args = GemvParams,
    dtype = F16,
    step = false,
    stage_emit = r#"
    if (has_bias != 0) {
        run_gemv_canonical_core<Policy, true>(matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes);
    } else {
        run_gemv_canonical_core<Policy, false>(matrix, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, scale_bytes);
    }"#
)]
pub struct GemvCanonical {
    #[arg(stage_skip)]
    pub matrix: TensorArg,
    #[arg(stage_skip)]
    pub scale_bytes: TensorArg,
    pub vector_x: TensorArg,
    #[arg(output)]
    pub result_y: TensorArg,
    pub params: GemvParams,
    pub bias: TensorArg,
    pub residual: TensorArg,
    pub alpha: f32,
    pub beta: f32,
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

    pub fn dispatch_config(&self) -> DispatchConfig {
        const COLS_PER_TG: usize = 8; // Match Legacy cols8 variant
        const TG_WIDTH: usize = 256; // 8 warps * 32 threads
        let n = self.params.n as usize;
        let batch = self.params.batch.max(1) as usize;
        let num_tgs = (n + COLS_PER_TG - 1) / COLS_PER_TG;
        DispatchConfig {
            grid: GridSize::new(num_tgs, 1, batch),
            group: ThreadgroupSize::d1(TG_WIDTH),
        }
    }
}

/// Kernel ID for pipeline caching.
pub struct GemvCanonicalId;

#[cfg(test)]
mod tests {}
