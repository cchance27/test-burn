use crate::{
    Context, MetalError, TensorElement, tensor::{CanonicalF16Tensor, Dtype, Tensor}
};

pub struct TransformerBlock<T: TensorElement> {
    // Attention weights (placeholders matching GGUF shapes)
    pub attn_qkv_weight: Tensor<T>,
    pub attn_qkv_weight_canon: Option<CanonicalF16Tensor<T>>,
    pub attn_qkv_bias: Tensor<T>,
    pub attn_out_weight: Tensor<T>,
    pub attn_out_weight_canon: Option<CanonicalF16Tensor<T>>,
    /// Optional packed Q8_0 weight for the attention output projection ([d_model, d_model]).
    pub attn_out_weight_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    /// Optional packed Q8_0 weight for the Q projection (row-major [d_model, d_model]).
    pub attn_q_weight_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    /// Optional packed Q8_0 weight for the K projection ([kv_dim, d_model]).
    pub attn_k_weight_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    /// Optional packed Q8_0 weight for the V projection ([kv_dim, d_model]).
    pub attn_v_weight_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,

    // =====================================================
    // NOTE: All dense weights are now transposed during loading via
    // copy_weight_transposed_into_fused, producing [In, Out] layout.
    // No separate transposed copies are needed.
    // attn_out_weight_transposed kept for backward compat with legacy path.
    // =====================================================
    /// Transposed attention output weight [d_model, d_model] - legacy only
    pub attn_out_weight_transposed: Option<Tensor<T>>,

    // Feedforward
    pub ffn_down: Tensor<T>,
    pub ffn_down_canon: Option<CanonicalF16Tensor<T>>,
    /// Optional packed Q8_0 weight for FFN down projection ([ff_dim, d_model]) or transpose-compatible.
    pub ffn_down_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_gate_up_weight: Tensor<T>,
    /// Optional packed Q8_0 weights for separate FFN gate/up projections.
    pub ffn_gate_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_up_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_gate: Tensor<T>,
    pub ffn_up: Tensor<T>,
    pub ffn_gate_canon: Option<CanonicalF16Tensor<T>>,
    pub ffn_up_canon: Option<CanonicalF16Tensor<T>>,
    // Biases for the FFN projections
    pub ffn_gate_bias: Tensor<T>,
    pub ffn_up_bias: Tensor<T>,
    pub ffn_down_bias: Tensor<T>,
    pub ffn_norm_gamma: Tensor<T>,

    // Pre-normalization before attention
    pub attn_norm_gamma: Tensor<T>,

    pub kv_dim: usize,
}

impl<T> TransformerBlock<T>
where
    T: TensorElement,
{
    pub fn new(cfg: &super::Qwen25Config, ctx: &mut Context<T>) -> Result<Self, MetalError> {
        let kv_dim = cfg.d_model * cfg.n_kv_heads / cfg.n_heads;

        // Q, K, V projections packed into a single fused matrix stored in row-major layout
        let qkv_out_dim = cfg.d_model + 2 * kv_dim;
        let attn_qkv_weight = Tensor::zeros(vec![cfg.d_model, qkv_out_dim], ctx, false)?;
        let attn_qkv_bias = Tensor::zeros(vec![qkv_out_dim], ctx, false)?;

        let attn_out_weight = Tensor::zeros(vec![cfg.d_model, cfg.d_model], ctx, false)?;

        let (attn_qkv_weight_canon, attn_out_weight_canon) = if T::DTYPE == Dtype::F16 {
            (
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, qkv_out_dim], ctx)?),
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.d_model], ctx)?),
            )
        } else {
            (None, None)
        };

        // FFN (SwiGLU)
        // Allocate FFN weights in [In, Out] = [d_model, ff_dim] layout (matching QKV)
        // This allows transposed loading to work correctly with transpose_right=false
        let ffn_down = Tensor::zeros(vec![cfg.ff_dim, cfg.d_model], ctx, false)?;
        // NOTE: ffn_gate_up_weight kept for backward compat but not used in unified path
        let ffn_gate_up_weight = Tensor::zeros(vec![cfg.d_model, 2 * cfg.ff_dim], ctx, false)?;
        // Allocate separate gate/up with [In, Out] dims for transposed loading
        let ffn_gate = Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx, false)?;
        let ffn_up = Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx, false)?;

        let (ffn_gate_canon, ffn_up_canon, ffn_down_canon) = if T::DTYPE == Dtype::F16 {
            (
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.ff_dim], ctx)?),
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.ff_dim], ctx)?),
                Some(CanonicalF16Tensor::new(vec![cfg.ff_dim, cfg.d_model], ctx)?),
            )
        } else {
            (None, None, None)
        };

        // FFN biases
        let ffn_gate_bias = Tensor::zeros(vec![cfg.ff_dim], ctx, false)?;
        let ffn_up_bias = Tensor::zeros(vec![cfg.ff_dim], ctx, false)?;
        let ffn_down_bias = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        // Norms
        let ffn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        let attn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        Ok(Self {
            attn_qkv_weight,
            attn_qkv_weight_canon,
            attn_qkv_bias,
            attn_out_weight,
            attn_out_weight_canon,
            attn_out_weight_q8: None,
            attn_q_weight_q8: None,
            attn_k_weight_q8: None,
            attn_v_weight_q8: None,
            // Transposed weights: none needed since all weights now transposed during load
            attn_out_weight_transposed: None,
            ffn_down,
            ffn_down_canon,
            ffn_down_q8: None,
            ffn_gate_up_weight,
            ffn_gate_q8: None,
            ffn_up_q8: None,
            ffn_gate,
            ffn_up,
            ffn_gate_canon,
            ffn_up_canon,
            ffn_gate_bias,
            ffn_up_bias,
            ffn_down_bias,
            ffn_norm_gamma,
            attn_norm_gamma,
            kv_dim,
        })
    }
}
