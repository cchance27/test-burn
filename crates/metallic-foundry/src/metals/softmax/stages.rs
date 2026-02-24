//! Softmax V2 Stages - Properly Composed
//!
//! Each stage handles ONLY its own concern:
//! - SoftmaxMaxStage: Computes local_max per thread
//! - SoftmaxSumStage: Computes local_sum per thread  
//! - SoftmaxNormStage: Normalizes and writes output
//!
//! SIMD reductions are handled by separate SimdStage instances.

use metallic_macros::Stage;

use crate::types::TensorArg;

/// Stage that computes per-thread local maximum.
/// Must be followed by SimdStage::reduce_max() to get row_max.
#[derive(Stage, Clone, Debug)]
#[stage(
    includes("dtypes/runtime_types.metal", "softmax/softmax.metal"),
    emit = r#"
    // Phase 1: Find local max per thread
    float local_max = find_row_max(matrix, row_idx, tid, seq_k, causal, mask_idx);
"#,
    out_var = "local_max"
)]
pub struct SoftmaxMaxStage {
    #[arg(buffer = 0, metal_type = "const device InputStorageT*")]
    pub matrix: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub mask_idx: u32,
}

impl SoftmaxMaxStage {
    pub fn new(_input_var: &str) -> Self {
        Self {
            matrix: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            mask_idx: 0,
        }
    }
}

/// Stage that computes per-thread exp sum.
/// Must be followed by SimdStage::reduce_sum() to get row_sum.
#[derive(Stage, Clone, Debug)]
#[stage(
    includes("dtypes/runtime_types.metal", "softmax/softmax.metal"),
    emit = r#"
    // Phase 2: Compute local exp sum per thread
    float local_sum = compute_exp_sum(matrix, row_max, row_idx, tid, seq_k, causal, mask_idx);
"#,
    out_var = "local_sum"
)]
pub struct SoftmaxSumStage {
    #[arg(buffer = 0, metal_type = "const device InputStorageT*")]
    pub matrix: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub mask_idx: u32,
}

impl SoftmaxSumStage {
    pub fn new(_max_var: &str) -> Self {
        Self {
            matrix: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            mask_idx: 0,
        }
    }
}

/// Stage that normalizes values and writes to output.
#[derive(Stage, Clone, Debug)]
#[stage(
    includes("dtypes/runtime_types.metal", "softmax/softmax.metal"),
    emit = r#"
    // Phase 3: Normalize and write output
    normalize_and_write(output, matrix, row_max, row_sum, row_idx, tid, seq_k, causal, mask_idx);
"#,
    out_var = "void"
)]
pub struct SoftmaxNormStage {
    #[arg(buffer = 2, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 0, metal_type = "const device InputStorageT*")]
    pub matrix: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub mask_idx: u32,
}

impl SoftmaxNormStage {
    pub fn new(_max_var: &str, _sum_var: &str) -> Self {
        Self {
            output: TensorArg::default(),
            matrix: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            mask_idx: 0,
        }
    }
}
