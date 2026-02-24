//! QKV Fused Stages - Componentized N-way dot product for QKV fusion.

use std::sync::Arc;

use metallic_macros::Stage as DeriveStage;

use crate::{fusion::MetalPolicy, metals::gemv::stages::VectorWidth, types::TensorArg};

/// Parallel projection stage that computes Q, K, and V projections in a single pass.
///
/// This stage maximizes register reuse by loading the input vector `x` once
/// and performing multiple dot products against different weight matrices.
///
/// It supports GQA by allowing different N dimensions for K and V.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes(
        "gemv/common.metal",
        "gemv/dot.metal",
        "gemv/vectorized_stage.metal",
        "gemv/scalar_output.metal",
        "qkv/qkv_project.metal"
    ),
    policy_field = "policy",
    template_bindings(
        vec_width = "self.vector_width.elements()",
        has_norm = "if self.norm_shared_name.is_some() { \"true\" } else { \"false\" }",
        norm_var = "self.norm_shared_name.as_deref().unwrap_or(\"0.0f\")"
    ),
    emit = r#"
    float3 {out_var} = run_parallel_qkv_project_stage<{policy_struct}, {vec_width}, {has_norm}>(
        w_q, s_q, w_k, s_k, w_v, s_v, input,
        k_dim, n_dim, n_kv, weights_per_block,
        gamma, {norm_var},
        lane_id, row_idx, batch_idx
    );
"#,
    out_var = "qkv_partial"
)]
// DEBT: fields are consumed by `#[derive(Stage)]` codegen and Metal emission, not direct Rust reads.
#[allow(dead_code)]
pub struct ParallelProjectStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    w_q: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    s_q: TensorArg,
    #[arg(buffer = 2, metal_type = "const device uchar*")]
    w_k: TensorArg,
    #[arg(buffer = 3, metal_type = "const device uchar*")]
    s_k: TensorArg,
    #[arg(buffer = 4, metal_type = "const device uchar*")]
    w_v: TensorArg,
    #[arg(buffer = 5, metal_type = "const device uchar*")]
    s_v: TensorArg,
    #[arg(buffer = 6, metal_type = "const device InputStorageT*")]
    input: TensorArg,
    #[arg(buffer = 7)]
    k_dim: u32,
    #[arg(buffer = 8)]
    n_dim: u32,
    #[arg(buffer = 9)]
    n_kv: u32,
    #[arg(buffer = 10)]
    weights_per_block: u32,
    #[arg(buffer = 18, metal_type = "const device GammaStorageT*")]
    gamma: TensorArg,
    #[arg(stage_skip)]
    /// Policy for quantization (F16, Q8)
    policy: Arc<dyn MetalPolicy>,
    #[arg(stage_skip)]
    /// Vector width for loads (matches layout stride)
    vector_width: VectorWidth,
    #[arg(stage_skip)]
    /// Optional shared memory variable name for normalization (e.g. "tg_inv_rms")
    norm_shared_name: Option<String>,
}

impl ParallelProjectStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            w_q: TensorArg::default(),
            s_q: TensorArg::default(),
            w_k: TensorArg::default(),
            s_k: TensorArg::default(),
            w_v: TensorArg::default(),
            s_v: TensorArg::default(),
            input: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            n_kv: 0,
            weights_per_block: 0,
            gamma: TensorArg::default(),
            policy,
            vector_width: VectorWidth::Vec8, // Default safe
            norm_shared_name: None,
        }
    }

    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }

    pub fn with_norm(mut self, shared_name: &str) -> Self {
        self.norm_shared_name = Some(shared_name.to_string());
        self
    }
}

/// Reduction stage for multiple partial sums (Q, K, V).
#[derive(Debug, Clone, Default, DeriveStage)]
#[stage(
    includes("qkv/qkv_project.metal"),
    emit = r#"
    float3 {out_var} = run_qkv_reduce_stage({input_var});
"#,
    out_var = "qkv_final"
)]
pub struct MultiWarpReduceStage;

/// Specialized write stage for QKV fused output.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("qkv/qkv_project.metal"),
    emit = r#"
    run_qkv_write_stage({input_var}, out_q, out_k, out_v, b_q, b_k, b_v, has_b, n_dim, n_kv, lane_id, row_idx, batch_idx);
"#,
    out_var = "void"
)]
// DEBT: fields are consumed by `#[derive(Stage)]` codegen and Metal emission, not direct Rust reads.
#[allow(dead_code)]
pub struct MultiWriteOutputStage {
    #[arg(buffer = 11, output, metal_type = "device OutputStorageT*")]
    out_q: TensorArg,
    #[arg(buffer = 12, output, metal_type = "device OutputStorageT*")]
    out_k: TensorArg,
    #[arg(buffer = 13, output, metal_type = "device OutputStorageT*")]
    out_v: TensorArg,
    #[arg(buffer = 14, metal_type = "const device BiasStorageT*")]
    b_q: TensorArg,
    #[arg(buffer = 15, metal_type = "const device BiasStorageT*")]
    b_k: TensorArg,
    #[arg(buffer = 16, metal_type = "const device BiasStorageT*")]
    b_v: TensorArg,
    #[arg(buffer = 17)]
    has_b: u32,
}

impl Default for MultiWriteOutputStage {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiWriteOutputStage {
    pub fn new() -> Self {
        Self {
            out_q: TensorArg::default(),
            out_k: TensorArg::default(),
            out_v: TensorArg::default(),
            b_q: TensorArg::default(),
            b_k: TensorArg::default(),
            b_v: TensorArg::default(),
            has_b: 0,
        }
    }
}
