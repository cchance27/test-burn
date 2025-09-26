use super::{Context, MetalError, Tensor, swiglu};
use crate::gguf::model_loader::GGUFModel;
use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddOp;
use crate::metallic::kernels::kv_rearrange::KvRearrangeOp;
use crate::metallic::kernels::matmul::MatMulOp;
use crate::metallic::kernels::rmsnorm::RMSNormOp;
use crate::metallic::kernels::rope::RoPEOp;
use crate::metallic::kernels::silu::SiluOp;
use crate::metallic::model::LoadableModel;

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
    pub fn new(cfg: &Qwen25Config, ctx: &mut Context) -> Result<Self, MetalError> {
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

        ctx.synchronize();

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

            // Debug for first token
            //if i == 0 {
            //    let debug_end = std::cmp::min(src_start + 10, embed_data.len());
            //    println!(
            //        "Weight for first token {} ({}): {:?}",
            //        token_id,
            //        src_start,
            //        &embed_data[src_start..debug_end]
            //    );
            //}
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
            let ffn_output_flat = swiglu::swiglu(
                &x_normed_mlp_flat,
                &block.ffn_gate,
                &block.ffn_gate_bias,
                &block.ffn_up,
                &block.ffn_up_bias,
                &block.ffn_down,
                &block.ffn_down_bias,
                ctx,
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

/// Implement the LoadableModel trait so Qwen25 can be created from a GGUFModel
impl LoadableModel for Qwen25 {
    fn load_from_gguf(gguf_model: &GGUFModel, ctx: &mut Context) -> Result<Self, MetalError> {
        use crate::gguf::GGUFValue;

        // Build config heuristically from metadata (fallbacks kept consistent with qwen25::new)
        let d_model = gguf_model.get_metadata_u32_or(&["qwen2.d_model", "model.d_model"], 896) as usize;
        let ff_dim = gguf_model.get_metadata_u32_or(&["qwen2.ff_dim", "model.ff_dim"], 4864) as usize;
        let n_heads = gguf_model.get_metadata_u32_or(&["qwen2.n_heads", "model.n_heads"], 14) as usize;
        let n_kv_heads = gguf_model.get_metadata_u32_or(&["qwen2.n_kv_heads", "model.n_kv_heads"], 2) as usize;
        let n_layers = gguf_model.get_metadata_u32_or(&["qwen2.n_layers", "model.n_layers"], 24) as usize;
        let seq_len = gguf_model.get_metadata_u32_or(&["qwen2.context_length", "model.context_length"], 32768) as usize;
        let rope_freq_base = gguf_model.get_metadata_f32_or(&["qwen2.rope_theta", "qwen2.rope.freq_base"], 1_000_000.0);
        let rms_eps = gguf_model.get_metadata_f32_or(&["qwen2.rms_norm_eps", "qwen2.rope.rms_eps"], 1e-6);

        // Determine vocabulary size:
        // 1) Prefer explicit `vocab_size` or `model.vocab_size` metadata
        // 2) Otherwise, if GGUF provides `tokenizer.ggml.tokens` (an array), use its length
        // 3) Fallback to legacy default (32000)
        let vocab_size = if let Some(GGUFValue::U32(v)) = gguf_model
            .metadata
            .entries
            .get("vocab_size")
            .or_else(|| gguf_model.metadata.entries.get("model.vocab_size"))
        {
            *v as usize
        } else if let Some(GGUFValue::Array(arr)) = gguf_model.metadata.entries.get("tokenizer.ggml.tokens") {
            // Use tokenizer token count when available to avoid token-id out-of-range errors.
            arr.len()
        } else {
            151936usize // vocab_size from qwen25 config.json
        };

        let cfg = Qwen25Config {
            n_layers,
            d_model,
            ff_dim,
            n_heads,
            n_kv_heads,
            seq_len,
            vocab_size,
            rope_freq_base,
            rms_eps,
        };

        // Instantiate Qwen25 with default-initialized weights
        let mut qwen = Qwen25::new(cfg, ctx)?;

        // Helper: attempt to copy from gguf tensor into dst.
        // Fast-path when shapes match, and a linear fallback when only total element counts match.
        // We avoid transposing rectangular matrices since GGUF already has the correct layout for our matmuls.
        fn try_copy(src: &crate::metallic::Tensor, dst: &mut crate::metallic::Tensor) -> Result<(), MetalError> {
            // Basic size check
            if src.len() != dst.len() {
                return Err(MetalError::DimensionMismatch {
                    expected: dst.len(),
                    actual: src.len(),
                });
            }

            // Exact-shape fast path
            if src.dims == dst.dims {
                let s = src.as_slice();
                let d = dst.as_mut_slice();
                d.copy_from_slice(s);
                return Ok(());
            }

            // Fallback linear copy for other shapes
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            d.copy_from_slice(s);
            Ok(())
        }

        // layer index extractor (searches for common patterns)
        fn parse_layer_index(name: &str) -> Option<usize> {
            let patterns = ["layers.", "layer.", "blocks.", "block.", "layer_", "layers_"];
            let lname = name.to_lowercase();
            for pat in patterns.iter() {
                if let Some(pos) = lname.find(pat) {
                    let start = pos + pat.len();
                    let mut digits = String::new();
                    for ch in lname[start..].chars() {
                        if ch.is_ascii_digit() {
                            digits.push(ch);
                        } else {
                            break;
                        }
                    }
                    if !digits.is_empty()
                        && let Ok(idx) = digits.parse::<usize>()
                    {
                        return Some(idx);
                    }
                }
            }
            // fallback: first run of digits
            let mut found = String::new();
            for ch in lname.chars() {
                if ch.is_ascii_digit() {
                    found.push(ch);
                } else if !found.is_empty() {
                    break;
                }
            }
            if !found.is_empty()
                && let Ok(idx) = found.parse::<usize>()
            {
                return Some(idx);
            }
            None
        }

        // Iterate gguf tensors and map into Qwen25 where names match heuristics.
        // This mapping is intentionally permissive to handle different exporter naming schemes.
        for (name, tensor) in &gguf_model.tensors {
            let lname = name.to_lowercase();

            // Embedding and weight tying for token embeddings
            if ((lname.contains("token") && lname.contains("emb")) || lname.contains("tok_emb") || lname.contains("tokembedding"))
                && tensor.len() == qwen.embed_weight.len()
            {
                // For this model, the data is already laid out for [vocab, d_model] despite GGUF dims; linear copy
                let src_slice = tensor.as_slice();
                let dst_slice = qwen.embed_weight.as_mut_slice();
                dst_slice.copy_from_slice(src_slice);
                //println!("MAPPING DEBUG: Linear copy for token_embd to [vocab, d_model]");

                continue;
            }

            // Output / lm_head
            if (lname.contains("lm_head") || lname.contains("lmhead") || (lname.contains("output") && lname.contains("weight")))
                && tensor.len() == qwen.output_weight.len()
            {
                try_copy(tensor, &mut qwen.output_weight).expect("succesfull copy");
                continue;
            }

            // Final norm (output_norm)
            if lname.contains("output_norm")
                || lname.contains("final_norm")
                || lname == "norm.weight" && tensor.len() == qwen.final_norm_gamma.len()
            {
                try_copy(tensor, &mut qwen.final_norm_gamma).expect("successful copy");
                continue;
            }

            // Handle weight tying - if this is the token embedding and we haven't set output weights yet,
            // use it as the output weight
            if lname.contains("token_embd") && lname.contains("weight") {
                // Check if output_weight is still all zeros (not set)
                let output_slice = qwen.output_weight.as_slice();
                let is_output_unset = output_slice.iter().all(|&x| x == 0.0);

                if is_output_unset {
                    // The GGUF token_embd.weight has shape [d_model, vocab_size] which matches our output_weight shape
                    if tensor.len() == qwen.output_weight.len() {
                        match try_copy(tensor, &mut qwen.output_weight) {
                            Ok(()) => {
                                //println!("MAPPING -> Applied weight tying: token_embd.weight -> output_weight");
                            }
                            Err(e) => {
                                println!("MAPPING -> token_embd.weight copy to output_weight failed: {:?}", e);
                            }
                        }
                    } else {
                        println!(
                            "MAPPING -> token_embd.weight size mismatch: {} vs {}",
                            tensor.len(),
                            qwen.output_weight.len()
                        );
                    }
                }
                continue;
            }

            // Per-layer parameters
            if let Some(layer_idx) = parse_layer_index(&lname) {
                if layer_idx >= qwen.blocks.len() {
                    continue;
                }
                let block = &mut qwen.blocks[layer_idx];

                // Attention projections (many exporters use different names)
                // Query
                if (lname.contains("wq")
                    || lname.contains("attn.q")
                    || lname.contains("attn_q")
                    || lname.contains("q_proj.weight")
                    || lname.contains("query.weight")
                    || lname.contains("q.weight")
                    || lname.contains("attention.query.weight"))
                    && tensor.len() == block.attn_q_weight.len()
                {
                    try_copy(tensor, &mut block.attn_q_weight).expect("succesfull copy");
                    continue;
                }

                // Key
                if (lname.contains("wk")
                    || lname.contains("attn.k")
                    || lname.contains("attn_k")
                    || lname.contains("k_proj.weight")
                    || lname.contains("key.weight")
                    || lname.contains("k.weight")
                    || lname.contains("attention.key.weight"))
                    && tensor.len() == block.attn_k_weight.len()
                {
                    try_copy(tensor, &mut block.attn_k_weight).expect("successful copy");
                    continue;
                }

                // Value
                if (lname.contains("wv")
                    || lname.contains("attn.v")
                    || lname.contains("attn_v")
                    || lname.contains("v_proj.weight")
                    || lname.contains("value.weight")
                    || lname.contains("v.weight")
                    || lname.contains("attention.value.weight"))
                    && tensor.len() == block.attn_v_weight.len()
                {
                    try_copy(tensor, &mut block.attn_v_weight).expect("succesfull copy");
                    continue;
                }
                // Attention output / projection (wo, outproj, o_proj)
                if (lname.contains("wo")
                    || lname.contains("attn_out")
                    || lname.contains("attn.out")
                    || lname.contains("out_proj")
                    || lname.contains("o.weight")
                    || lname.contains("out.weight")
                    || lname.contains("attention.output.weight"))
                    && tensor.len() == block.attn_out_weight.len()
                {
                    try_copy(tensor, &mut block.attn_out_weight).expect("succesfull copy");
                    continue;
                }

                // Attention biases (optional)
                if (lname.contains("attn.q.bias") || lname.contains("attn_q.bias") || lname.contains("attention.query.bias"))
                    && tensor.len() == block.attn_q_bias.len()
                {
                    try_copy(tensor, &mut block.attn_q_bias).expect("succesfull copy");
                    continue;
                }

                if (lname.contains("attn.k.bias") || lname.contains("attn_k.bias") || lname.contains("attention.key.bias"))
                    && tensor.len() == block.attn_k_bias.len()
                {
                    try_copy(tensor, &mut block.attn_k_bias).expect("succesfull copy");
                    continue;
                }
                if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
                    && tensor.len() == block.attn_v_bias.len()
                {
                    try_copy(tensor, &mut block.attn_v_bias).expect("succesfull copy");
                    continue;
                }
                if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
                    && tensor.len() == block.attn_v_bias.len()
                {
                    try_copy(tensor, &mut block.attn_v_bias).expect("succesfull copy");
                    continue;
                }

                // FFN down/gate/up (common aliases) - Updated for correct SwiGLU mapping
                // gate_proj (for SiLU activation)
                if (lname.contains("mlp.gate_proj.weight")
                    || lname.contains("ffn.gate")
                    || lname.contains("ffn_gate")
                    || lname.contains("gate.weight")
                    || lname.contains("wg.weight")
                    || lname.contains("w1.weight"))
                    && tensor.len() == block.ffn_gate.len()
                {
                    if tensor.dims == block.ffn_gate.dims() {
                        try_copy(tensor, &mut block.ffn_gate).expect("successful copy for ffn_gate (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_gate.dims().len() == 2
                        && tensor.dims[0] == block.ffn_gate.dims()[1]
                        && tensor.dims[1] == block.ffn_gate.dims()[0]
                    {
                        // GGUF metadata reports dims swapped, but data is already laid out row-major in
                        // [ff_dim, d_model]; copy directly without transposing to preserve values.
                        let src = tensor.as_slice();
                        let dst = block.ffn_gate.as_mut_slice();
                        dst.copy_from_slice(src);
                    } else {
                        try_copy(tensor, &mut block.ffn_gate).expect("successful linear copy fallback for ffn_gate");
                    }
                    continue;
                }

                // up_proj (for element-wise multiplication)
                if (lname.contains("mlp.up_proj.weight")
                    || lname.contains("ffn.up")
                    || lname.contains("ffn_up")
                    || lname.contains("up.weight")
                    || lname.contains("w3.weight"))
                    && tensor.len() == block.ffn_up.len()
                {
                    if tensor.dims == block.ffn_up.dims() {
                        try_copy(tensor, &mut block.ffn_up).expect("successful copy for ffn_up (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_up.dims().len() == 2
                        && tensor.dims[0] == block.ffn_up.dims()[1]
                        && tensor.dims[1] == block.ffn_up.dims()[0]
                    {
                        let src = tensor.as_slice();
                        let dst = block.ffn_up.as_mut_slice();
                        dst.copy_from_slice(src);
                    } else {
                        try_copy(tensor, &mut block.ffn_up).expect("successful linear copy fallback for ffn_up");
                    }
                    continue;
                }

                // down_proj (final projection)
                if (lname.contains("mlp.down_proj.weight")
                    || lname.contains("ffn.down")
                    || lname.contains("ffn_down")
                    || lname.contains("down.weight")
                    || lname.contains("w2.weight")
                    || lname.contains("fc2.weight")
                    || lname.contains("wo.weight"))
                    && tensor.len() == block.ffn_down.len()
                {
                    if tensor.dims == block.ffn_down.dims() {
                        try_copy(tensor, &mut block.ffn_down).expect("successful copy for ffn_down (exact match)");
                    } else if tensor.dims.len() == 2
                        && block.ffn_down.dims().len() == 2
                        && tensor.dims[0] == block.ffn_down.dims()[1]
                        && tensor.dims[1] == block.ffn_down.dims()[0]
                    {
                        let src = tensor.as_slice();
                        let dst = block.ffn_down.as_mut_slice();
                        dst.copy_from_slice(src);
                        //println!("MAPPING DEBUG: FFN down loaded via direct copy with swapped dims metadata");
                    } else {
                        try_copy(tensor, &mut block.ffn_down).expect("successful linear copy fallback for ffn_down");
                    }
                    continue;
                }

                // FFN biases (gate/up/down) - map biases into block fields when present
                if lname.contains("mlp.gate_proj.bias")
                    || lname.contains("ffn.gate.bias")
                    || lname.contains("ffn_gate.bias")
                    || lname.contains("gate.bias")
                    || lname.contains("w1.bias")
                    || lname.contains("wg.bias")
                {
                    if tensor.len() == block.ffn_gate_bias.len() {
                        try_copy(tensor, &mut block.ffn_gate_bias).expect("successful copy of gate bias");
                    }
                    continue;
                }
                if lname.contains("mlp.up_proj.bias")
                    || lname.contains("ffn.up.bias")
                    || lname.contains("ffn_up.bias")
                    || lname.contains("up.bias")
                    || lname.contains("w3.bias")
                {
                    if tensor.len() == block.ffn_up_bias.len() {
                        try_copy(tensor, &mut block.ffn_up_bias).expect("successful copy of up bias");
                    }
                    continue;
                }
                if lname.contains("mlp.down_proj.bias")
                    || lname.contains("ffn.down.bias")
                    || lname.contains("ffn_down.bias")
                    || lname.contains("down.bias")
                    || lname.contains("w2.bias")
                    || lname.contains("b2.bias")
                    || lname.contains("fc2.bias")
                    || lname.contains("wo.bias")
                {
                    if tensor.len() == block.ffn_down_bias.len() {
                        try_copy(tensor, &mut block.ffn_down_bias).expect("successful copy of down bias");
                    }
                    continue;
                }

                // Norm gammas (RMSNorm)
                if ((lname.contains("attn_norm") && lname.contains("gamma"))
                    || lname.contains("attn.g")
                    || lname.contains("attn_norm.weight")
                    || lname.contains("attention.layernorm.weight")
                    || lname.contains("ln_attn.weight"))
                    && tensor.len() == block.attn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.attn_norm_gamma).expect("succesfull copy");
                    continue;
                }

                // (old duplicate FFN bias handling removed)

                // Norm gammas (RMSNorm)
                if ((lname.contains("attn_norm") && lname.contains("gamma"))
                    || lname.contains("attn.g")
                    || lname.contains("attn_norm.weight")
                    || lname.contains("attention.layernorm.weight")
                    || lname.contains("ln_attn.weight"))
                    && tensor.len() == block.attn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.attn_norm_gamma).expect("succesfull copy");
                    continue;
                }
                if ((lname.contains("proj_norm") && lname.contains("gamma"))
                    || lname.contains("proj.g")
                    || lname.contains("ffn_norm.weight")
                    || lname.contains("ffn.layernorm.weight")
                    || lname.contains("ln_ffn.weight"))
                    && tensor.len() == block.ffn_norm_gamma.len()
                {
                    try_copy(tensor, &mut block.ffn_norm_gamma).expect("succesfull copy");
                    continue;
                }
            }
        }

        // Synchronize to ensure all copies are complete
        ctx.synchronize();

        Ok(qwen)
    }
}
