use metallic_macros::{KernelArgs, Stage};

use crate::{
    compound::{CompiledCompoundKernel, stages::LayoutStage}, metals::common::cache::get_or_build_compound_kernel, types::TensorArg
};

const SOFTMAX_SDPA_BATCHED_METAL: &str = include_str!("softmax_sdpa_batched.metal");

#[derive(Stage, Clone, Debug)]
#[stage(
    emit = r#"
    // Phase 1: Find local max per thread (SDPA batched)
    float local_max = find_row_max_sdpa_batched(matrix, row_idx, tid, seq_k, causal, query_offset, rows_per_batch);
"#,
    out_var = "local_max",
    struct_defs_fn = "metal_defs"
)]
pub struct SoftmaxSdpaBatchedMaxStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub matrix: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub query_offset: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    pub rows_per_batch: u32,
}

impl SoftmaxSdpaBatchedMaxStage {
    pub fn new(_input_var: &str) -> Self {
        Self {
            matrix: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            query_offset: 0,
            rows_per_batch: 0,
        }
    }

    pub fn metal_defs() -> String {
        SOFTMAX_SDPA_BATCHED_METAL.to_string()
    }
}

#[derive(Stage, Clone, Debug)]
#[stage(
    emit = r#"
    // Phase 2: Compute local exp sum per thread (SDPA batched)
    float local_sum = compute_exp_sum_sdpa_batched(matrix, row_max, row_idx, tid, seq_k, causal, query_offset, rows_per_batch);
"#,
    out_var = "local_sum",
    struct_defs_fn = "metal_defs"
)]
pub struct SoftmaxSdpaBatchedSumStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub matrix: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub query_offset: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    pub rows_per_batch: u32,
}

impl SoftmaxSdpaBatchedSumStage {
    pub fn new(_max_var: &str) -> Self {
        Self {
            matrix: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            query_offset: 0,
            rows_per_batch: 0,
        }
    }

    pub fn metal_defs() -> String {
        SOFTMAX_SDPA_BATCHED_METAL.to_string()
    }
}

#[derive(Stage, Clone, Debug)]
#[stage(
    emit = r#"
    // Phase 3: Normalize and write output (SDPA batched)
    normalize_and_write_sdpa_batched(output, matrix, row_max, row_sum, row_idx, tid, seq_k, causal, query_offset, rows_per_batch);
"#,
    out_var = "void",
    struct_defs_fn = "metal_defs"
)]
pub struct SoftmaxSdpaBatchedNormStage {
    #[arg(buffer = 2, output, metal_type = "device half*")]
    pub output: TensorArg,
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub matrix: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub seq_k: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub causal: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    pub query_offset: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    pub rows_per_batch: u32,
}

impl SoftmaxSdpaBatchedNormStage {
    pub fn new(_max_var: &str, _sum_var: &str) -> Self {
        Self {
            output: TensorArg::default(),
            matrix: TensorArg::default(),
            seq_k: 0,
            causal: 0,
            query_offset: 0,
            rows_per_batch: 0,
        }
    }

    pub fn metal_defs() -> String {
        SOFTMAX_SDPA_BATCHED_METAL.to_string()
    }
}

pub fn get_softmax_v2_sdpa_batched_kernel() -> std::sync::Arc<CompiledCompoundKernel> {
    use crate::compound::stages::SimdStage;
    get_or_build_compound_kernel("softmax_sdpa_batched", "v2", || {
        crate::metals::common::composition::manual_output("softmax_v2_sdpa_batched")
            .prologue(LayoutStage::row_major())
            .prologue(SoftmaxSdpaBatchedMaxStage::new("matrix"))
            .prologue(SimdStage::reduce_max("local_max", "row_max"))
            .prologue(SoftmaxSdpaBatchedSumStage::new("row_max"))
            .prologue(SimdStage::reduce_sum("local_sum", "row_sum"))
            .main(SoftmaxSdpaBatchedNormStage::new("row_max", "row_sum"))
            .compile()
    })
}

#[derive(Debug, KernelArgs)]
pub struct SoftmaxV2SdpaBatchedArgs {
    #[arg(buffer = 0)]
    pub input: TensorArg,
    #[arg(buffer = 1)]
    pub scale: TensorArg,
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    #[arg(buffer = 3)]
    pub seq_k: u32,
    #[arg(buffer = 4)]
    pub causal: u32,
    #[arg(buffer = 5)]
    pub query_offset: u32,
    #[arg(buffer = 6)]
    pub rows_per_batch: u32,
}
