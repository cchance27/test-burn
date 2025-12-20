use crate::{
    Context, MetalError, TensorElement, tensor::{CanonicalF16Tensor, Dtype, Tensor}
};

pub struct TransformerBlock<T: TensorElement> {
    // Attention weights (placeholders matching GGUF shapes)
    pub attn_qkv_weight: Option<Tensor<T>>,
    // Canonical weights must be separate for Q, K, V to allow correct GEMV dispatch
    // because the canonical layout interleaves columns within K-blocks.
    pub attn_q_weight_canon: Option<CanonicalF16Tensor<T>>,
    pub attn_k_weight_canon: Option<CanonicalF16Tensor<T>>,
    pub attn_v_weight_canon: Option<CanonicalF16Tensor<T>>,
    pub attn_qkv_bias: Tensor<T>,
    pub attn_out_weight: Option<Tensor<T>>,
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
    pub ffn_down: Option<Tensor<T>>,
    pub ffn_down_canon: Option<CanonicalF16Tensor<T>>,
    /// Optional packed Q8_0 weight for FFN down projection ([ff_dim, d_model]) or transpose-compatible.
    pub ffn_down_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_gate_up_weight: Option<Tensor<T>>,
    /// Optional packed Q8_0 weights for separate FFN gate/up projections.
    pub ffn_gate_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_up_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub ffn_gate: Option<Tensor<T>>,
    pub ffn_up: Option<Tensor<T>>,
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
        let qkv_out_dim = cfg.d_model + 2 * kv_dim;
        let attn_qkv_bias = Tensor::zeros(vec![qkv_out_dim], ctx, false)?;

        // Q, K, V projections packed into a single fused matrix stored in row-major layout
        let (attn_qkv_weight, attn_out_weight, attn_q_weight_canon, attn_k_weight_canon, attn_v_weight_canon, attn_out_weight_canon) =
            if T::DTYPE == Dtype::F16 {
                (
                    None,
                    None,
                    Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.d_model], ctx)?),
                    Some(CanonicalF16Tensor::new(vec![cfg.d_model, kv_dim], ctx)?),
                    Some(CanonicalF16Tensor::new(vec![cfg.d_model, kv_dim], ctx)?),
                    Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.d_model], ctx)?),
                )
            } else {
                (
                    Some(Tensor::zeros(vec![cfg.d_model, qkv_out_dim], ctx, false)?),
                    Some(Tensor::zeros(vec![cfg.d_model, cfg.d_model], ctx, false)?),
                    None,
                    None,
                    None,
                    None,
                )
            };

        let (ffn_down, ffn_gate_up_weight, ffn_gate, ffn_up, ffn_gate_canon, ffn_up_canon, ffn_down_canon) = if T::DTYPE == Dtype::F16 {
            (
                None,
                None,
                None,
                None,
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.ff_dim], ctx)?),
                Some(CanonicalF16Tensor::new(vec![cfg.d_model, cfg.ff_dim], ctx)?),
                Some(CanonicalF16Tensor::new(vec![cfg.ff_dim, cfg.d_model], ctx)?),
            )
        } else {
            (
                Some(Tensor::zeros(vec![cfg.ff_dim, cfg.d_model], ctx, false)?),
                Some(Tensor::zeros(vec![cfg.d_model, 2 * cfg.ff_dim], ctx, false)?),
                Some(Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx, false)?),
                Some(Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx, false)?),
                None,
                None,
                None,
            )
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
            attn_q_weight_canon,
            attn_k_weight_canon,
            attn_v_weight_canon,
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
