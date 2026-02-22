//! GemvV2 Stages - Full-featured stages for GEMV using Stage composition.
//!
//! Features:
//! - Canonical 4x/8x unrolling for maximum throughput
//! - NK/KN layout support via WEIGHT_INDEX macro from LayoutStage
//! - Policy templates for transparent Q8/F16 dequantization
//! - Composable via CompoundKernel

use std::sync::Arc;

use metallic_macros::Stage as DeriveStage;

use crate::{fusion::MetalPolicy, policy::activation::Activation, types::TensorArg};

// =============================================================================
// VectorizedDotStage - Unified, quant-agnostic vectorized dot product
// =============================================================================

/// Vector load width for dot product stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorWidth {
    /// 4 elements per load (half4)
    Vec4,
    /// 8 elements per load (float4 reinterpreted as 8 halves)
    #[default]
    Vec8,
}

impl VectorWidth {
    /// Number of elements loaded per thread per iteration.
    pub fn elements(&self) -> u32 {
        match self {
            VectorWidth::Vec4 => 4,
            VectorWidth::Vec8 => 8,
        }
    }
}

/// Unified vectorized dot product stage for warp-per-row GEMV.
///
/// Features:
/// - Parameterized by `Arc<dyn MetalPolicy>` - policy-agnostic via trait
/// - Uses vectorized Policy::load_weights<8>() for maximum throughput
/// - Designed for use with `WarpLayoutStage` (warp-per-row dispatch)
/// - Each lane loads 8 elements per K chunk, all lanes cover 256 elements
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("gemv/gemv.metal"),
    policy_field = "policy",
    template_bindings(
        vec_width = "self.vector_width.elements()",
        use_f16_cols8 = "if self.f16_cols8 && self.policy.meta().address_unit_bytes == 2 && matches!(self.vector_width, VectorWidth::Vec8) { \"true\" } else { \"false\" }",
        has_gamma = "if self.use_gamma { \"true\" } else { \"false\" }",
        has_shared_norm = "if self.norm_shared_name.is_some() { \"true\" } else { \"false\" }",
        norm_var = "self.norm_shared_name.as_deref().unwrap_or(\"0.0f\")"
    ),
    emit = r#"
    float {out_var} = run_gemv_vectorized_stage<{policy_struct}, {vec_width}, {use_f16_cols8}>(
        weights,
        scale_bytes,
        input,
        residual,
        {norm_var},
        k_dim,
        n_dim,
        weights_per_block,
        row_idx,
        lane_id,
        batch_idx,
        lid.x,
        {has_gamma},
        {has_shared_norm}
    );
"#,
    out_var = "partial_dot"
)]
#[allow(dead_code)]
pub struct VectorizedDotStage {
    /// Policy for code generation (determines header, struct name, etc.)
    #[arg(skip, stage_skip)]
    policy: Arc<dyn MetalPolicy>,
    /// Vector width for loads
    #[arg(skip, stage_skip)]
    vector_width: VectorWidth,
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    scale_bytes: TensorArg,
    #[arg(buffer = 2, metal_type = "const device half*")]
    input: TensorArg,
    // Fixed ABI slot for optional fused RMS gamma.
    // In plain GEMV this aliases residual buffer and is ignored when `use_gamma=false`.
    #[arg(buffer = 10, metal_type = "const device half*")]
    residual: TensorArg,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    k_dim: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    n_dim: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    weights_per_block: u32,
    /// Optional shared memory variable name for normalization (e.g. "tg_inv_rms")
    #[arg(skip, stage_skip)]
    norm_shared_name: Option<String>,
    #[arg(skip, stage_skip)]
    use_gamma: bool,
    /// Use F16-specific cols8 SIMD path (RowMajor only).
    #[arg(skip, stage_skip)]
    f16_cols8: bool,
}

impl VectorizedDotStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            policy,
            vector_width: VectorWidth::Vec8,
            weights: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            input: TensorArg::default(),
            residual: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            weights_per_block: 0,
            norm_shared_name: None,
            use_gamma: false,
            f16_cols8: false,
        }
    }

    pub fn with_norm(mut self, shared_name: &str) -> Self {
        self.use_gamma = true;
        self.norm_shared_name = Some(shared_name.to_string());
        self
    }

    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }

    pub fn with_f16_cols8(mut self, enabled: bool) -> Self {
        self.f16_cols8 = enabled;
        self
    }
}

/// Canonical dot product stage (Legacy V1 compatibility).
///
/// Uses `gemv_dot_canonical` which implements a 4-way unrolled loop.
/// This acts as a robust fallback or alternative to the vectorized stage.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("gemv/gemv.metal"),
    policy_field = "policy",
    emit = r#"
    float {out_var} = run_gemv_canonical_stage<{policy_struct}>(
        weights,
        scale_bytes,
        input,
        row_idx,
        lane_id,
        batch_idx,
        k_dim,
        n_dim,
        weights_per_block
    );
"#,
    out_var = "partial_dot"
)]
#[allow(dead_code)]
pub struct CanonicalDotStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    scale_bytes: TensorArg,
    #[arg(buffer = 2, metal_type = "const device half*")]
    input: TensorArg,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    k_dim: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    n_dim: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    weights_per_block: u32,
    #[arg(skip, stage_skip)]
    policy: Arc<dyn MetalPolicy>,
}

impl CanonicalDotStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            weights: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            input: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            weights_per_block: 0,
            policy,
        }
    }
}

/// Stage that writes the reduced result to output with optional bias.
/// Designed for warp-per-row dispatch where only lane 0 writes.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    activation_field = "activation",
    emit = r#"
    // Write output (only lane 0 of each warp)
    if (lane_id == 0) {
        // Apply alpha scaling to the reduced sum
        float scaled_sum = row_sum * alpha;
        float result = scaled_sum;
        if (has_bias != 0) {
            result += (float)bias[row_idx];
        }
        
        // Apply activation
        result = {activation_struct}::apply(result);

        if (has_residual != 0) {
            result += ((float)residual[batch_idx * n_dim + row_idx]) * beta;
        }
        output[batch_idx * n_dim + row_idx] = (half)result;
    }
"#,
    out_var = "void"
)]
pub struct WarpWriteOutputStage {
    #[arg(buffer = 3, output)]
    pub output: TensorArg,
    #[arg(buffer = 7, metal_type = "const device half*")]
    pub bias: TensorArg,
    #[arg(buffer = 8, metal_type = "constant uint&")]
    pub has_bias: u32,
    #[arg(buffer = 9, metal_type = "constant float&")]
    pub alpha: f32,
    #[arg(buffer = 10, metal_type = "const device half*")]
    pub residual: TensorArg,
    #[arg(buffer = 11, metal_type = "constant uint&")]
    pub has_residual: u32,
    #[arg(buffer = 12, metal_type = "constant float&")]
    pub beta: f32,
    #[arg(skip, stage_skip)]
    pub activation: Activation,
}

impl WarpWriteOutputStage {
    pub fn new() -> Self {
        Self {
            output: TensorArg::default(),
            bias: TensorArg::default(),
            has_bias: 0,
            alpha: 1.0,
            residual: TensorArg::default(),
            has_residual: 0,
            beta: 1.0,
            activation: Activation::None,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Default for WarpWriteOutputStage {
    fn default() -> Self {
        Self::new()
    }
}

/// Stage that writes the reduced result to output with optional bias (no residual).
///
/// This is used by kernels that already consume higher buffer slots (e.g. fused RMSNorm paths)
/// and don't support/need residual accumulation in the write stage.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    activation_field = "activation",
    emit = r#"
    // Write output (only lane 0 of each warp)
    if (lane_id == 0) {
        float result = row_sum * alpha;
        if (has_bias != 0) {
            result += (float)bias[row_idx];
        }

        // Apply activation
        result = {activation_struct}::apply(result);

        output[batch_idx * n_dim + row_idx] = (half)result;
    }
"#,
    out_var = "void"
)]
pub struct WarpWriteOutputNoResidualStage {
    #[arg(buffer = 3, output)]
    pub output: TensorArg,
    #[arg(buffer = 7, metal_type = "const device half*")]
    pub bias: TensorArg,
    #[arg(buffer = 8, metal_type = "constant uint&")]
    pub has_bias: u32,
    #[arg(buffer = 9, metal_type = "constant float&")]
    pub alpha: f32,
    #[arg(skip, stage_skip)]
    pub activation: Activation,
}

impl WarpWriteOutputNoResidualStage {
    pub fn new() -> Self {
        Self {
            output: TensorArg::default(),
            bias: TensorArg::default(),
            has_bias: 0,
            alpha: 1.0,
            activation: Activation::None,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Default for WarpWriteOutputNoResidualStage {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ScalarDotStage - Thread-per-row dot product (Large N optimized)
// =============================================================================

/// Dot product stage for thread-per-row dispatch.
///
/// Designed for Layout::ColMajor (KxN weights) where N is the contiguous dimension.
/// Threads in a warp access adjacent weights (W[k, n], W[k, n+1]...), enabling
/// full memory coalescing without implicit vector loads.
#[derive(Debug, Clone, DeriveStage)]
#[stage(
    includes("gemv/gemv.metal"),
    policy_field = "policy",
    template_bindings(unroll = "self.unroll.max(1)"),
    emit = r#"
    // Scalar Dot Product ({policy_struct})
    float {out_var} = run_gemv_scalar_dot_stage<{policy_struct}, {unroll}>(
        weights,
        scale_bytes,
        input,
        row_idx,
        k_dim,
        n_dim,
        weights_per_block
    );
"#,
    out_var = "row_sum"
)]
#[allow(dead_code)]
pub struct ScalarDotStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    weights: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    scale_bytes: TensorArg,
    #[arg(buffer = 2, metal_type = "const device half*")]
    input: TensorArg,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    k_dim: u32,
    #[arg(buffer = 5, metal_type = "constant uint&")]
    n_dim: u32,
    #[arg(buffer = 6, metal_type = "constant uint&")]
    weights_per_block: u32,
    #[arg(skip, stage_skip)]
    policy: Arc<dyn MetalPolicy>,
    #[arg(skip, stage_skip)]
    unroll: usize,
}

impl ScalarDotStage {
    pub fn new(policy: Arc<dyn MetalPolicy>) -> Self {
        Self {
            weights: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            input: TensorArg::default(),
            k_dim: 0,
            n_dim: 0,
            weights_per_block: 0,
            policy,
            unroll: 8,
        }
    }
}
