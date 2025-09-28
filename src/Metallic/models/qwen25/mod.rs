use crate::gguf::model_loader::GGUFModel;
use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
use crate::metallic::instrumentation::{LatencyEvent, MemoryEvent};
use crate::metallic::kernels::kv_rearrange::KvRearrangeOp;
use crate::metallic::kernels::matmul::MatMulOp;
use crate::metallic::kernels::repeat_kv_heads::RepeatKvHeadsOp;
use crate::metallic::kernels::rmsnorm::RMSNormOp;
use crate::metallic::kernels::rope::RoPEOp;
use crate::metallic::kernels::silu::SiluOp;
use crate::metallic::models::LoadableModel;
use crate::metallic::{Context, MetalError, Tensor};
use std::time::Instant;

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
    pub rope_cos_cache: Tensor,
    pub rope_sin_cache: Tensor,
}

impl Qwen25 {
    /// Embed tokens into d_model dimensional vectors
    pub fn embed(&self, tokens: &[u32], ctx: &mut Context) -> Result<Tensor, MetalError> {
        let batch = 1; // For now, assume batch size of 1
        let seq = tokens.len();

        // Create output tensor [batch, seq, d_model]
        let mut embedded = Tensor::zeros(vec![batch, seq, self.config.d_model], ctx, true)?;

        // Get the embedding weight data
        let embed_data = self.embed_weight.as_slice();

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

        // Reshape for matrix multiplication: [batch*seq, d_model]
        let m = batch * seq;
        let flat_hidden = hidden.reshape(vec![m, d_model])?;

        // Apply output projection: [batch*seq, d_model] x [vocab_size, d_model].T -> [batch*seq, vocab_size]
        let logits_flat = ctx.matmul(&flat_hidden, &self.output_weight, false, true)?;

        // Synchronize to ensure matmul is complete before reading values

        // Reshape back to [batch, seq, vocab_size]
        let logits = logits_flat.reshape(vec![batch, seq, self.config.vocab_size])?;

        Ok(logits)
    }

    /// Forward pass that takes tokens as input and returns logits
    pub fn forward_tokens(&self, tokens: &[u32], ctx: &mut Context) -> Result<Tensor, MetalError> {
        // Embed tokens
        let embedded = self.embed(tokens, ctx)?;

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

        // Apply output projection
        let logits = self.output(&hidden, ctx)?;

        Ok(logits)
    }

    pub fn new(config: Qwen25Config, ctx: &mut Context) -> Result<Self, MetalError> {
        // allocate embed and output weights
        let embed_weight = Tensor::zeros(vec![config.vocab_size, config.d_model], ctx, false)?;
        let output_weight = Tensor::zeros(vec![config.vocab_size, config.d_model], ctx, false)?;
        let final_norm_gamma = Tensor::zeros(vec![config.d_model], ctx, false)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(&config, ctx)?);
        }

        // Pre-compute RoPE frequencies
        // NOTE: This assumes head_dim for Q and K are the same, which is true for Qwen2.5-0.5B
        let head_dim = config.d_model / config.n_heads;
        let dim_half = head_dim / 2;
        // These RoPE caches are retained for the lifetime of the model, so allocate them
        // outside of the transient memory pool to survive `Context::reset_pool()` calls.
        let mut cos_cache = Tensor::zeros(vec![config.seq_len, dim_half], ctx, false)?;
        let mut sin_cache = Tensor::zeros(vec![config.seq_len, dim_half], ctx, false)?;
        let cos_slice = cos_cache.as_mut_slice();
        let sin_slice = sin_cache.as_mut_slice();

        for pos in 0..config.seq_len {
            for i in 0..dim_half {
                let idx = pos * dim_half + i;
                let exponent = (2 * i) as f32 / head_dim as f32;
                let inv_freq = 1.0f32 / config.rope_freq_base.powf(exponent);
                let angle = pos as f32 * inv_freq;
                cos_slice[idx] = angle.cos();
                sin_slice[idx] = angle.sin();
            }
        }

        Ok(Self {
            config,
            blocks,
            embed_weight,
            output_weight,
            final_norm_gamma,
            rope_cos_cache: cos_cache,
            rope_sin_cache: sin_cache,
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

            // QKV GEMMs
            let m = batch * seq;
            let kv_dim = block.kv_dim;
            let x_flat = x_normed_attn.reshape(vec![m, d_model])?;
            let (q_mat, k_mat, v_mat) = ctx.fused_qkv_projection(&x_flat, &block.attn_qkv_weight, &block.attn_qkv_bias, d_model, kv_dim)?;

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

            // Apply RoPE per head on Q and K using head_dim (and kv_head_dim)
            let q_heads_after_rope = ctx.call::<RoPEOp>((
                q_heads,
                self.rope_cos_cache.clone(),
                self.rope_sin_cache.clone(),
                head_dim as u32,
                seq as u32,
                0,
            ))?;
            let k_heads_after_rope = ctx.call::<RoPEOp>((
                k_heads,
                self.rope_cos_cache.clone(),
                self.rope_sin_cache.clone(),
                kv_head_dim as u32,
                seq as u32,
                0,
            ))?;

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

            // Residual Add
            x = resid_attn.add_elem(&attn_out, ctx)?;

            // --- MLP Block ---
            let resid_mlp = x.clone();

            // RMSNorm before MLP
            let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32))?;
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
            let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;

            // Residual Add
            x = resid_mlp.add_elem(&ffn_output, ctx)?;
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32))?;

        Ok(final_normed)
    }

    /// Step-forward for autoregressive generation with KV caching.
    pub fn forward_step(&self, input: &Tensor, pos: usize, ctx: &mut Context) -> Result<Tensor, MetalError> {
        // Validate input shape: expect [batch, 1, d_model]
        let dims = input.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "qwen25::forward_step expects input shape [batch, 1, d_model], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1]; // seq is always 1
        let d_model = dims[2];

        let mut x = input.clone();
        let overall_start = Instant::now();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_start = Instant::now();
            let resid_attn = x.clone();

            ctx.record_memory_event(MemoryEvent::BlockStart { index: layer_idx });

            // RMSNorm before Attention
            let mut phase_start = Instant::now();
            let x_normed_attn = ctx.call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32))?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "attn_norm"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "attn_norm"));

            // QKV GEMMs for the single token
            let m = batch * seq; // m is always 1 for a single token
            let kv_dim = block.kv_dim;
            let x_flat = x_normed_attn.reshape(vec![m, d_model])?;

            phase_start = Instant::now();
            let (q_mat, k_mat, v_mat) = ctx.fused_qkv_projection(&x_flat, &block.attn_qkv_weight, &block.attn_qkv_bias, d_model, kv_dim)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "attn_qkv_proj"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "attn_qkv_proj"));

            // KV Head Rearrangement
            let n_heads = self.config.n_heads;
            let n_kv_heads = self.config.n_kv_heads;
            let head_dim = d_model / n_heads;
            let kv_head_dim = kv_dim / n_kv_heads;

            phase_start = Instant::now();
            let q_heads = ctx.call::<KvRearrangeOp>((
                q_mat,
                d_model as u32,
                head_dim as u32,
                n_heads as u32,
                n_heads as u32,
                head_dim as u32,
                seq as u32,
            ))?;
            let k_heads = ctx.call::<KvRearrangeOp>((
                k_mat,
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
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "attn_rearrange"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "attn_rearrange"));

            // Apply RoPE using the pre-computed cache for the current position
            let position_offset = pos as u32;
            phase_start = Instant::now();
            let q_heads_after_rope = ctx.call::<RoPEOp>((
                q_heads,
                self.rope_cos_cache.clone(),
                self.rope_sin_cache.clone(),
                head_dim as u32,
                seq as u32,
                position_offset,
            ))?;
            let k_heads_after_rope = ctx.call::<RoPEOp>((
                k_heads,
                self.rope_cos_cache.clone(),
                self.rope_sin_cache.clone(),
                kv_head_dim as u32,
                seq as u32,
                position_offset,
            ))?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "rope"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "rope"));

            // Update the KV cache with the new K and V values
            phase_start = Instant::now();
            ctx.write_kv_step(layer_idx, pos, &k_heads_after_rope, &v_heads)?;

            // Retrieve the full K and V caches for attention
            let (k_cache, v_cache, _) = ctx
                .kv_caches
                .get(&layer_idx)
                .cloned()
                .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not found", layer_idx)))?;
            let k_history = Qwen25::gather_cache_history(&k_cache, pos + 1, ctx)?;
            let v_history = Qwen25::gather_cache_history(&v_cache, pos + 1, ctx)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "kv_cache"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "kv_cache"));

            // Repeat K and V to match Q head count for GQA
            let group_size = n_heads / n_kv_heads;
            phase_start = Instant::now();
            let k_repeated = Qwen25::repeat_kv_heads(&k_history, group_size, batch, n_kv_heads, n_heads, pos + 1, kv_head_dim, ctx)?;
            let v_repeated = Qwen25::repeat_kv_heads(&v_history, group_size, batch, n_kv_heads, n_heads, pos + 1, kv_head_dim, ctx)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "kv_repeat"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "kv_repeat"));

            // SDPA (causal mask enabled)
            phase_start = Instant::now();
            let attn_out_heads = ctx.scaled_dot_product_attention_with_offset(&q_heads_after_rope, &k_repeated, &v_repeated, true, pos)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "sdpa"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "sdpa"));

            // Attention Output Reassembly
            phase_start = Instant::now();
            let attn_out_reshaped_1 = attn_out_heads.reshape(vec![batch, n_heads, seq, head_dim])?;
            let attn_out_permuted = attn_out_reshaped_1.permute(&[0, 2, 1, 3], ctx)?;
            let attn_out_reshaped = attn_out_permuted.reshape(vec![batch, seq, d_model])?;

            let attn_out = ctx
                .matmul(&attn_out_reshaped.reshape(vec![m, d_model])?, &block.attn_out_weight, false, true)?
                .reshape(vec![batch, seq, d_model])?;

            // Residual Add
            x = resid_attn.add_elem(&attn_out, ctx)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "attn_output"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "attn_output"));

            // --- MLP Block ---
            let resid_mlp = x.clone();
            phase_start = Instant::now();
            let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32))?;
            let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model])?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "mlp_norm"), phase_start.elapsed());

            phase_start = Instant::now();
            let ffn_output_flat = ctx.SwiGLU(
                &x_normed_mlp_flat,
                &block.ffn_gate,
                &block.ffn_gate_bias,
                &block.ffn_up,
                &block.ffn_up_bias,
                &block.ffn_down,
                &block.ffn_down_bias,
            )?;
            let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "mlp_swiglu"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "mlp_swiglu"));

            // Residual Add
            phase_start = Instant::now();
            x = resid_mlp.add_elem(&ffn_output, ctx)?;
            ctx.record_latency_event(LatencyEvent::block_phase(layer_idx, "mlp_output"), phase_start.elapsed());
            ctx.record_memory_event(MemoryEvent::block_phase(layer_idx, "mlp_output"));

            ctx.record_latency_event(LatencyEvent::Block { index: layer_idx }, block_start.elapsed());
            ctx.record_memory_event(MemoryEvent::BlockEnd { index: layer_idx });
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32))?;

        ctx.record_latency_event(LatencyEvent::ForwardStep, overall_start.elapsed());
        ctx.record_memory_event(MemoryEvent::ForwardSample);

        Ok(final_normed)
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

        ctx.call::<RepeatKvHeadsOp>((
            input.clone(),
            group_size as u32,
            batch as u32,
            n_kv_heads as u32,
            n_heads as u32,
            seq as u32,
            head_dim as u32,
        ))
    }

    fn gather_cache_history(cache: &Tensor, steps: usize, ctx: &mut Context) -> Result<Tensor, MetalError> {
        let dims = cache.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "KV cache tensor must have shape [seq, batch_heads, head_dim]".to_string(),
            ));
        }
        if steps == 0 || steps > dims[0] {
            return Err(MetalError::InvalidShape(format!(
                "Requested {} KV steps exceeds cache capacity {}",
                steps, dims[0]
            )));
        }

        #[allow(clippy::single_range_in_vec_init)]
        let mut cache_view = cache.slice(&[0..steps])?;
        ctx.prepare_tensors_for_active_cmd(&mut [&mut cache_view]);

        cache_view.permute(&[1, 0, 2], ctx)
    }
}
