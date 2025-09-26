use crate::gguf::model_loader::GGUFModel;
use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddOp;
use crate::metallic::kernels::kv_rearrange::KvRearrangeOp;
use crate::metallic::kernels::matmul::MatMulOp;
use crate::metallic::kernels::rmsnorm::RMSNormOp;
use crate::metallic::kernels::rope::RoPEOp;
use crate::metallic::kernels::silu::SiluOp;
use crate::metallic::models::LoadableModel;
use crate::metallic::{Context, MetalError, Tensor};

mod qwen25_tests;

mod transformer_block;
use transformer_block::TransformerBlock;

mod loading;

/// Qwen25 configuration derived from Qwen2.5 metadata.
/// Matches Qwen2.5-Coder-0.5B: d_model=896, ff_dim=4864, n_heads=14, n_kv_heads=2, n_layers=24.
pub struct Qwen25Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub ff_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
}

pub struct Qwen25 {
    pub config: Qwen25Config,
    pub blocks: Vec<TransformerBlock>,
    pub embed_weight: Tensor,
    pub output_weight: Tensor,
    pub final_norm_gamma: Tensor,
}

impl Qwen25 {
    /// Embed tokens into d_model dimensional vectors
    pub fn embed(&self, tokens: &[u32], ctx: &mut Context) -> Result<Tensor, MetalError> {
        let batch = 1; // For now, assume batch size of 1
        let seq = tokens.len();

        // Create output tensor [batch, seq, d_model]
        let mut embedded = Tensor::zeros(vec![batch, seq, self.config.d_model], ctx)?;

        // Get the embedding weight data
        let embed_data = self.embed_weight.as_slice();
        ctx.synchronize();

        // For each token, look up its embedding
        let output_data = embedded.as_mut_slice();
        for (i, &token_id) in tokens.iter().enumerate() {
            if token_id as usize >= self.config.vocab_size {
                return Err(MetalError::InvalidShape(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    token_id, self.config.vocab_size
                )));
            }

            // Copy the embedding for this token
            let src_start = (token_id as usize) * self.config.d_model;
            let src_end = src_start + self.config.d_model;
            let dst_start = i * self.config.d_model;
            let dst_end = dst_start + self.config.d_model;

            output_data[dst_start..dst_end].copy_from_slice(&embed_data[src_start..src_end]);
        }

        Ok(embedded)
    }

    /// Apply the output layer to convert from d_model to vocab_size
    pub fn output(&self, hidden: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        // Validate input shape: expect [batch, seq, d_model]
        let dims = hidden.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(format!(
                "qwen25.output expects input with 3 dims [batch, seq, d_model], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1];
        let d_model = dims[2];
        if d_model != self.config.d_model {
            return Err(MetalError::InvalidShape(format!(
                "Input d_model {} does not match config.d_model {}",
                d_model, self.config.d_model
            )));
        }

        ctx.synchronize();

        // Reshape for matrix multiplication: [batch*seq, d_model]
        let m = batch * seq;
        let flat_hidden = hidden.reshape(vec![m, d_model])?;
        ctx.synchronize();

        // Apply output projection: [batch*seq, d_model] x [vocab_size, d_model].T -> [batch*seq, vocab_size]
        let logits_flat = ctx.matmul(&flat_hidden, &self.output_weight, false, true)?;

        // Synchronize to ensure matmul is complete before reading values
        ctx.synchronize();

        // Reshape back to [batch, seq, vocab_size]
        let logits = logits_flat.reshape(vec![batch, seq, self.config.vocab_size])?;

        Ok(logits)
    }

    /// Forward pass that takes tokens as input and returns logits
    pub fn forward_tokens(&self, tokens: &[u32], ctx: &mut Context) -> Result<Tensor, MetalError> {
        // Embed tokens
        let embedded = self.embed(tokens, ctx)?;
        ctx.synchronize();

        let embedded_slice = embedded.as_slice();
        let embedded_max = embedded_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let embedded_min = embedded_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("Embedded stats - max: {:.4}, min: {:.4}", embedded_max, embedded_min);

        // Additional debug: sample first few values
        if embedded_slice.len() >= 5 {
            println!(
                "First 5 embedded values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
                embedded_slice[0], embedded_slice[1], embedded_slice[2], embedded_slice[3], embedded_slice[4]
            );
        }

        println!("Forward tokens: processing {} tokens", tokens.len());

        // Run through transformer blocks
        let hidden = self.forward(&embedded, ctx)?;

        // Synchronize to ensure forward pass is complete before reading values
        ctx.synchronize();

        // Apply output projection
        let logits = self.output(&hidden, ctx)?;
        ctx.synchronize();

        Ok(logits)
    }

    pub fn new(config: Qwen25Config, ctx: &mut Context) -> Result<Self, MetalError> {
        // allocate embed and output weights
        let embed_weight = Tensor::zeros(vec![config.vocab_size, config.d_model], ctx)?;
        let output_weight = Tensor::zeros(vec![config.vocab_size, config.d_model], ctx)?;
        let final_norm_gamma = Tensor::zeros(vec![config.d_model], ctx)?;
        ctx.synchronize();

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(&config, ctx)?);
        }

        Ok(Self {
            config,
            blocks,
            embed_weight,
            output_weight,
            final_norm_gamma,
        })
    }

    pub fn forward(&self, input: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        // Validate input shape: expect [batch, seq, d_model]
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(format!(
                "qwen25.forward expects input with 3 dims [batch, seq, d_model], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1];
        let d_model = dims[2];
        if d_model != self.config.d_model {
            return Err(MetalError::InvalidShape(format!(
                "Input d_model {} does not match config.d_model {}",
                d_model, self.config.d_model
            )));
        }
        if seq > self.config.seq_len {
            return Err(MetalError::InvalidShape(format!(
                "Input seq {} exceeds configured seq_len {}",
                seq, self.config.seq_len
            )));
        }

        let mut x = input.clone();

        for block in self.blocks.iter() {
            let resid_attn = x.clone();

            // RMSNorm before Attention
            let x_normed_attn = ctx.call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32))?;
            ctx.synchronize();

            // QKV GEMMs
            let m = batch * seq;
            let kv_dim = block.attn_k_weight.dims()[0];
            let x_flat = x_normed_attn.reshape(vec![m, d_model])?;

            let q_temp = ctx.matmul(&x_flat, &block.attn_q_weight, false, true)?;
            // Ensure matmul is finished before we read values for CPU sanity checks
            ctx.synchronize();

            let q_mat = ctx.call::<BroadcastElemwiseAddOp>((q_temp, block.attn_q_bias.clone()))?;
            ctx.synchronize(); // Ensure Q matmul and bias add complete

            let k_temp = ctx.matmul(&x_flat, &block.attn_k_weight, false, true)?;
            ctx.synchronize();

            let k_mat = ctx.call::<BroadcastElemwiseAddOp>((k_temp, block.attn_k_bias.clone()))?;
            ctx.synchronize(); // Ensure K matmul and bias add complete

            let v_temp = ctx.matmul(&x_flat, &block.attn_v_weight, false, true)?;
            ctx.synchronize();

            let v_mat = ctx.call::<BroadcastElemwiseAddOp>((v_temp, block.attn_v_bias.clone()))?;
            ctx.synchronize(); // Ensure V matmul and bias add complete

            // Defer RoPE until after head rearrangement
            let (q_after, k_after) = (q_mat.clone(), k_mat.clone());
            // KV Head Rearrangement
            let n_heads = self.config.n_heads;
            let n_kv_heads = self.config.n_kv_heads;
            let head_dim = d_model / n_heads;
            let kv_head_dim = kv_dim / n_kv_heads;

            let q_heads = ctx.call::<KvRearrangeOp>((
                q_after,
                d_model as u32,
                head_dim as u32,
                n_heads as u32,
                n_heads as u32,
                head_dim as u32,
                seq as u32,
            ))?;
            let k_heads = ctx.call::<KvRearrangeOp>((
                k_after,
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ))?;
            let v_heads = ctx.call::<KvRearrangeOp>((
                v_mat,
                kv_dim as u32,
                kv_head_dim as u32,
                n_kv_heads as u32,
                n_kv_heads as u32,
                kv_head_dim as u32,
                seq as u32,
            ))?;
            ctx.synchronize();

            // Apply RoPE per head on Q and K using head_dim (and kv_head_dim)
            let q_heads_after_rope = {
                let dim_half = head_dim / 2;
                let mut cos_buf = vec![0f32; seq * dim_half];
                let mut sin_buf = vec![0f32; seq * dim_half];
                for pos in 0..seq {
                    for i in 0..dim_half {
                        let idx = pos * dim_half + i;
                        let exponent = (2 * i) as f32 / head_dim as f32;
                        let inv_freq = 1.0f32 / self.config.rope_freq_base.powf(exponent);
                        let angle = pos as f32 * inv_freq;
                        cos_buf[idx] = angle.cos();
                        sin_buf[idx] = angle.sin();
                    }
                }
                let cos = Tensor::create_tensor_from_slice(&cos_buf, vec![seq, dim_half], ctx)?;
                let sin = Tensor::create_tensor_from_slice(&sin_buf, vec![seq, dim_half], ctx)?;

                // Use the new kernel system
                ctx.call::<RoPEOp>((q_heads, cos, sin, head_dim as u32, seq as u32))?
            };
            ctx.synchronize();

            let k_heads_after_rope = {
                let dim_half = kv_head_dim / 2;
                let mut cos_buf = vec![0f32; seq * dim_half];
                let mut sin_buf = vec![0f32; seq * dim_half];
                for pos in 0..seq {
                    for i in 0..dim_half {
                        let idx = pos * dim_half + i;
                        let exponent = (2 * i) as f32 / kv_head_dim as f32;
                        let inv_freq = 1.0f32 / self.config.rope_freq_base.powf(exponent);
                        let angle = pos as f32 * inv_freq;
                        cos_buf[idx] = angle.cos();
                        sin_buf[idx] = angle.sin();
                    }
                }

                let cos = Tensor::create_tensor_from_slice(&cos_buf, vec![seq, dim_half], ctx)?;
                let sin = Tensor::create_tensor_from_slice(&sin_buf, vec![seq, dim_half], ctx)?;
                ctx.synchronize();

                // Use the new kernel system
                ctx.call::<RoPEOp>((k_heads, cos, sin, kv_head_dim as u32, seq as u32))?
            };
            // Repeat K and V to match Q head count for SDPA (GQA)
            let group_size = n_heads / n_kv_heads;

            let k_repeated = Qwen25::repeat_kv_heads(&k_heads_after_rope, group_size, batch, n_kv_heads, n_heads, seq, kv_head_dim, ctx)?;

            let v_repeated = Qwen25::repeat_kv_heads(&v_heads, group_size, batch, n_kv_heads, n_heads, seq, kv_head_dim, ctx)?;

            // SDPA (causal mask enabled)
            let attn_out_heads = ctx.scaled_dot_product_attention(&q_heads_after_rope, &k_repeated, &v_repeated, true)?;

            // Attention Output Reassembly
            let attn_out_reshaped = attn_out_heads
                .reshape(vec![batch, n_heads, seq, head_dim])?
                .permute(&[0, 2, 1, 3], ctx)?
                .reshape(vec![batch, seq, d_model])?;

            let attn_out = ctx
                .matmul(
                    &attn_out_reshaped.reshape(vec![m, d_model])?,
                    &block.attn_out_weight,
                    false,
                    true, // Transpose the output weight for correct dimensions
                )?
                .reshape(vec![batch, seq, d_model])?;
            ctx.synchronize(); // Ensure attention output matmul complete

            // Residual Add
            x = resid_attn.add_elem(&attn_out, ctx)?;
            ctx.synchronize(); // Ensure residual add complete

            // --- MLP Block ---
            let resid_mlp = x.clone();

            // RMSNorm before MLP
            let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32))?;
            ctx.synchronize();
            let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model])?;

            // FFN using extracted SwiGLU
            let ffn_output_flat = ctx.SwiGLU(
                &x_normed_mlp_flat,
                &block.ffn_gate,
                &block.ffn_gate_bias,
                &block.ffn_up,
                &block.ffn_up_bias,
                &block.ffn_down,
                &block.ffn_down_bias,
            )?;
            ctx.synchronize();
            let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;
            ctx.synchronize(); // Ensure SwiGLU operation complete

            // Residual Add
            x = resid_mlp.add_elem(&ffn_output, ctx)?;
            ctx.synchronize(); // Ensure final residual add complete
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32))?;
        ctx.synchronize();

        Ok(final_normed)
    }

    /// Step-forward for autoregressive generation. For now this is a thin wrapper
    /// around `forward` that validates the input has sequence length 1. We'll
    /// replace this with an incremental on-device KV-aware implementation next.
    pub fn step_forward(&self, input: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        let dims = input.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "qwen25::step_forward expects input shape [batch, 1, d_model], got {:?}",
                dims
            )));
        }

        // For autoregressive generation, we need to implement proper KV caching
        // For now, we'll use the full forward pass but in the future we'll optimize this
        self.forward(input, ctx)
    }

    /// Repeat KV heads for GQA to match Q head count
    #[allow(clippy::too_many_arguments)]
    fn repeat_kv_heads(
        input: &Tensor,
        group_size: usize,
        batch: usize,
        n_kv_heads: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        ctx: &mut Context,
    ) -> Result<Tensor, MetalError> {
        let input_dims = input.dims();
        if input_dims.len() != 3 || input_dims[0] != batch * n_kv_heads || input_dims[1] != seq || input_dims[2] != head_dim {
            return Err(MetalError::InvalidShape("Invalid input dimensions for repeat_kv_heads".to_string()));
        }

        let output_dims = vec![batch * n_heads, seq, head_dim];
        let mut output = Tensor::zeros(output_dims, ctx)?;

        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        for b in 0..batch {
            for h_kv in 0..n_kv_heads {
                let input_offset_base = ((b * n_kv_heads + h_kv) * seq) * head_dim;
                for g in 0..group_size {
                    let h = h_kv * group_size + g;
                    let output_offset_base = ((b * n_heads + h) * seq) * head_dim;
                    for s in 0..seq {
                        let input_offset = input_offset_base + s * head_dim;
                        let output_offset = output_offset_base + s * head_dim;
                        let src = &input_slice[input_offset..input_offset + head_dim];
                        let dst = &mut output_slice[output_offset..output_offset + head_dim];
                        dst.copy_from_slice(src);
                    }
                }
            }
        }

        Ok(output)
    }
}
