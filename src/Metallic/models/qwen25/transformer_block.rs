use crate::metallic::{Context, MetalError, Tensor};

pub struct TransformerBlock {
    // Attention weights (placeholders matching GGUF shapes)
    pub attn_q_weight: Tensor,
    pub attn_q_bias: Tensor,
    pub attn_k_weight: Tensor,
    pub attn_k_bias: Tensor,
    pub attn_v_weight: Tensor,
    pub attn_v_bias: Tensor,
    pub attn_out_weight: Tensor,

    // Feedforward
    pub ffn_down: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    // Biases for the FFN projections
    pub ffn_gate_bias: Tensor,
    pub ffn_up_bias: Tensor,
    pub ffn_down_bias: Tensor,
    pub ffn_norm_gamma: Tensor,

    // Pre-normalization before attention
    pub attn_norm_gamma: Tensor,
}

impl TransformerBlock {
    pub fn new(cfg: &super::Qwen25Config, ctx: &mut Context) -> Result<Self, MetalError> {
        // Q, K, V projections
        let attn_q_weight = Tensor::zeros(vec![cfg.d_model, cfg.d_model], ctx)?;
        let attn_q_bias = Tensor::zeros(vec![cfg.d_model], ctx)?;

        let kv_dim = cfg.d_model * cfg.n_kv_heads / cfg.n_heads;
        let attn_k_weight = Tensor::zeros(vec![kv_dim, cfg.d_model], ctx)?;
        let attn_k_bias = Tensor::zeros(vec![kv_dim], ctx)?;

        let attn_v_weight = Tensor::zeros(vec![kv_dim, cfg.d_model], ctx)?;
        let attn_v_bias = Tensor::zeros(vec![kv_dim], ctx)?;

        let attn_out_weight = Tensor::zeros(vec![cfg.d_model, cfg.d_model], ctx)?;

        // FFN (SwiGLU)
        // Allocate FFN weights in the layout expected by `swiglu`:
        // - gate/up: [d_model, ff_dim]
        // - down:    [ff_dim, d_model]
        let ffn_down = Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx)?;
        let ffn_gate = Tensor::zeros(vec![cfg.ff_dim, cfg.d_model], ctx)?;
        let ffn_up = Tensor::zeros(vec![cfg.ff_dim, cfg.d_model], ctx)?;

        // FFN biases
        let ffn_gate_bias = Tensor::zeros(vec![cfg.ff_dim], ctx)?;
        let ffn_up_bias = Tensor::zeros(vec![cfg.ff_dim], ctx)?;
        let ffn_down_bias = Tensor::zeros(vec![cfg.d_model], ctx)?;

        // Norms
        let ffn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx)?;

        let attn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx)?;

        Ok(Self {
            attn_q_weight,
            attn_q_bias,
            attn_k_weight,
            attn_k_bias,
            attn_v_weight,
            attn_v_bias,
            attn_out_weight,
            ffn_down,
            ffn_gate,
            ffn_up,
            ffn_gate_bias,
            ffn_up_bias,
            ffn_down_bias,
            ffn_norm_gamma,
            attn_norm_gamma,
        })
    }
}
