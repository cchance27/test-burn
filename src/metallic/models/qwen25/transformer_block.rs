use crate::metallic::tensor::Tensor;
use crate::metallic::{Context, MetalError, TensorElement};

pub struct TransformerBlock<T: TensorElement> {
    // Attention weights (placeholders matching GGUF shapes)
    pub attn_qkv_weight: Tensor<T>,
    pub attn_qkv_bias: Tensor<T>,
    pub attn_out_weight: Tensor<T>,

    // Feedforward
    pub ffn_down: Tensor<T>,
    pub ffn_gate_up_weight: Tensor<T>,
    pub ffn_gate: Tensor<T>,
    pub ffn_up: Tensor<T>,
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

        // FFN (SwiGLU)
        // Allocate FFN weights in the layout expected by `swiglu`:
        // - gate/up: [d_model, ff_dim]
        // - down:    [ff_dim, d_model]
        let ffn_down = Tensor::zeros(vec![cfg.d_model, cfg.ff_dim], ctx, false)?;
        let ffn_gate_up_weight = Tensor::zeros(vec![2 * cfg.ff_dim, cfg.d_model], ctx, false)?;
        let ffn_gate = ffn_gate_up_weight.slice(&[0..cfg.ff_dim])?;
        let ffn_up = ffn_gate_up_weight.slice(&[cfg.ff_dim..2 * cfg.ff_dim])?;

        // FFN biases
        let ffn_gate_bias = Tensor::zeros(vec![cfg.ff_dim], ctx, false)?;
        let ffn_up_bias = Tensor::zeros(vec![cfg.ff_dim], ctx, false)?;
        let ffn_down_bias = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        // Norms
        let ffn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        let attn_norm_gamma = Tensor::zeros(vec![cfg.d_model], ctx, false)?;

        Ok(Self {
            attn_qkv_weight,
            attn_qkv_bias,
            attn_out_weight,
            ffn_down,
            ffn_gate_up_weight,
            ffn_gate,
            ffn_up,
            ffn_gate_bias,
            ffn_up_bias,
            ffn_down_bias,
            ffn_norm_gamma,
            attn_norm_gamma,
            kv_dim,
        })
    }
}
