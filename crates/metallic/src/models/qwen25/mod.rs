use crate::kernels::kv_rearrange::KvRearrangeOp;
use crate::kernels::repeat_kv_heads::RepeatKvHeadsOp;
use crate::kernels::rmsnorm::RMSNormOp;
use crate::kernels::rope::RoPEOp;
use crate::{Context, MetalError, Tensor, TensorElement};
use metallic_instrumentation::{MetricEvent, record_metric_async};
use std::time::{Duration, Instant};

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

pub struct Qwen25<T: TensorElement> {
    pub config: Qwen25Config,
    pub blocks: Vec<TransformerBlock<T>>,
    pub embed_weight: Tensor<T>,
    pub output_weight: Tensor<T>,
    pub final_norm_gamma: Tensor<T>,
    pub rope_cos_cache: Tensor<T>,
    pub rope_sin_cache: Tensor<T>,
}

impl<T: TensorElement> Qwen25<T> {
    /// Embed tokens into d_model dimensional vectors
    pub fn embed(&self, tokens: &[u32], ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
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
    pub fn output(&self, hidden: &Tensor<T>, ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
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
    pub fn forward_tokens(&self, tokens: &[u32], ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
        // Embed tokens
        let embedded = self.embed(tokens, ctx)?;

        let embedded_slice = embedded.as_slice();
        let embedded_max = embedded_slice.iter().map(|&v| T::to_f32(v)).fold(f32::NEG_INFINITY, f32::max);
        let embedded_min = embedded_slice.iter().map(|&v| T::to_f32(v)).fold(f32::INFINITY, f32::min);
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

    pub fn new(config: Qwen25Config, ctx: &mut Context<T>) -> Result<Self, MetalError> {
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
                cos_slice[idx] = T::from_f32(angle.cos());
                sin_slice[idx] = T::from_f32(angle.sin());
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

    pub fn forward(&self, input: &Tensor<T>, ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
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

            let k_history = CacheHistory::from_tensor(k_heads_after_rope)?;
            let v_history = CacheHistory::from_tensor(v_heads)?;

            let k_repeated = Qwen25::repeat_kv_heads(&k_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, ctx)?;

            let v_repeated = Qwen25::repeat_kv_heads(&v_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, ctx)?;

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
                Some(&block.ffn_gate_up_weight),
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
    pub fn forward_step(&self, input: &Tensor<T>, pos: usize, ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
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

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_start = Instant::now();
            x = ctx.with_gpu_scope(format!("block_{}", layer_idx), |ctx| -> Result<Tensor<T>, MetalError> {
                // Accumulate CPU time between GPU calls in this block
                let mut cpu_accum = Duration::ZERO;
                let mut cpu_chk = Instant::now();
                ctx.set_pending_gpu_scope(format!("attn_residual_clone_block_{}_op", layer_idx));
                let resid_attn = x.clone();

                // RMSNorm before Attention
                ctx.set_pending_gpu_scope(format!("attn_norm_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x_normed_attn = ctx.call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32))?;
                cpu_chk = Instant::now();

                // QKV GEMMs for the single token
                let m = batch * seq; // m is always 1 for a single token
                let kv_dim = block.kv_dim;
                let x_flat = x_normed_attn.reshape(vec![m, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                let (q_mat, k_mat, v_mat) = ctx.with_gpu_scope(format!("attn_qkv_proj_block_{}_op", layer_idx), |ctx| {
                    ctx.fused_qkv_projection(&x_flat, &block.attn_qkv_weight, &block.attn_qkv_bias, d_model, kv_dim)
                })?;
                cpu_chk = Instant::now();

                // KV Head Rearrangement
                let n_heads = self.config.n_heads;
                let n_kv_heads = self.config.n_kv_heads;
                let head_dim = d_model / n_heads;
                let kv_head_dim = kv_dim / n_kv_heads;

                cpu_accum += cpu_chk.elapsed();
                let (q_heads, k_heads, v_heads) = ctx.with_gpu_scope(format!("attn_rearrange_block_{}_op", layer_idx), |ctx| {
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
                    Ok::<_, MetalError>((q_heads, k_heads, v_heads))
                })?;
                cpu_chk = Instant::now();

                // Apply RoPE using the pre-computed cache for the current position
                let position_offset = pos as u32;
                cpu_accum += cpu_chk.elapsed();
                let (q_heads_after_rope, k_heads_after_rope) = ctx.with_gpu_scope(format!("rope_block_{}_op", layer_idx), |ctx| {
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
                    Ok::<_, MetalError>((q_heads_after_rope, k_heads_after_rope))
                })?;
                cpu_chk = Instant::now();

                // Update the KV cache with the new K and V values
                let group_size = n_heads / n_kv_heads;
                cpu_accum += cpu_chk.elapsed();
                ctx.with_gpu_scope(format!("kv_cache_block_{}_op", layer_idx), |ctx| {
                    ctx.write_kv_step(layer_idx, pos, group_size, &k_heads_after_rope, &v_heads)
                })?;
                cpu_chk = Instant::now();

                // Create a view over the repeated KV cache for attention
                let cache_entry = ctx
                    .kv_caches
                    .get(&layer_idx)
                    .cloned()
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not found", layer_idx)))?;
                let k_repeated_history = Qwen25::gather_cache_history(&cache_entry.k, pos + 1, ctx)?;
                let v_repeated_history = Qwen25::gather_cache_history(&cache_entry.v, pos + 1, ctx)?;
                cpu_accum += cpu_chk.elapsed();
                let (k_repeated, v_repeated) = ctx.with_gpu_scope(format!("kv_repeat_block_{}_op", layer_idx), |ctx| {
                    let k_repeated =
                        Qwen25::repeat_kv_heads(&k_repeated_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, ctx)?;
                    let v_repeated =
                        Qwen25::repeat_kv_heads(&v_repeated_history, group_size, batch, n_kv_heads, n_heads, kv_head_dim, ctx)?;
                    Ok::<_, MetalError>((k_repeated, v_repeated))
                })?;
                cpu_chk = Instant::now();

                // SDPA (causal mask enabled)
                cpu_accum += cpu_chk.elapsed();
                let attn_out_heads = ctx.with_gpu_scope(format!("sdpa_block_{}_op", layer_idx), |ctx| {
                    ctx.scaled_dot_product_attention_with_offset(&q_heads_after_rope, &k_repeated, &v_repeated, true, pos)
                })?;
                cpu_chk = Instant::now();

                // Attention Output Reassembly
                ctx.set_pending_gpu_scope(format!("attn_reassembly_block_{}_op", layer_idx));
                let attn_out_reshaped_1 = attn_out_heads.reshape(vec![batch, n_heads, seq, head_dim])?;
                let attn_out_permuted = attn_out_reshaped_1.permute(&[0, 2, 1, 3], ctx)?;
                let attn_out_reshaped = attn_out_permuted.reshape(vec![batch, seq, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                let attn_out = ctx
                    .with_gpu_scope(format!("attn_output_block_{}_op", layer_idx), |ctx| {
                        ctx.matmul(&attn_out_reshaped.reshape(vec![m, d_model])?, &block.attn_out_weight, false, true)
                    })?
                    .reshape(vec![batch, seq, d_model])?;
                cpu_chk = Instant::now();

                // Residual Add
                ctx.set_pending_gpu_scope(format!("attn_residual_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x = resid_attn.add_elem(&attn_out, ctx)?;
                cpu_chk = Instant::now();

                // --- MLP Block ---
                ctx.set_pending_gpu_scope(format!("mlp_residual_clone_block_{}_op", layer_idx));
                let resid_mlp = x.clone();
                ctx.set_pending_gpu_scope(format!("mlp_norm_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32))?;
                cpu_chk = Instant::now();
                let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                let ffn_output_flat = ctx.with_gpu_scope(format!("mlp_swiglu_block_{}_op", layer_idx), |ctx| {
                    ctx.SwiGLU(
                        &x_normed_mlp_flat,
                        &block.ffn_gate,
                        &block.ffn_gate_bias,
                        &block.ffn_up,
                        &block.ffn_up_bias,
                        &block.ffn_down,
                        &block.ffn_down_bias,
                        Some(&block.ffn_gate_up_weight),
                    )
                })?;
                cpu_chk = Instant::now();
                ctx.set_pending_gpu_scope(format!("mlp_reshape_block_{}_op", layer_idx));
                let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;

                // Residual Add
                ctx.set_pending_gpu_scope(format!("mlp_residual_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x = resid_mlp.add_elem(&ffn_output, ctx)?;
                //cpu_chk = Instant::now(); // Not needed as we've finished accumulation of cpu ops i believe

                // Record per-block CPU overhead outside GPU calls
                let cpu_us = cpu_accum.as_micros() as u64;
                if cpu_us > 0 {
                    record_metric_async!(MetricEvent::InternalKernelCompleted {
                        parent_op_name: "generation_loop".to_string(),
                        internal_kernel_name: format!("forward_cpu_block_{}", layer_idx),
                        duration_us: cpu_us,
                    });
                }
                Ok(x)
            })?;
            let block_duration = block_start.elapsed();
            if !block_duration.is_zero() {
                record_metric_async!(MetricEvent::InternalKernelCompleted {
                    parent_op_name: "generation_loop".to_string(),
                    internal_kernel_name: format!("block_{}_total", layer_idx),
                    duration_us: block_duration.as_micros() as u64,
                });
            }
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32))?;

        Ok(final_normed)
    }

    /// Repeat KV heads for GQA to match Q head count
    #[allow(clippy::too_many_arguments)]
    fn repeat_kv_heads(
        history: &CacheHistory<T>,
        group_size: usize,
        batch: usize,
        n_kv_heads: usize,
        n_heads: usize,
        head_dim: usize,
        ctx: &mut Context<T>,
    ) -> Result<Tensor<T>, MetalError> {
        if n_kv_heads == 0 || n_heads == 0 {
            return Err(MetalError::InvalidShape("Invalid head counts for repeat_kv_heads".to_string()));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(MetalError::InvalidShape("Invalid head counts for repeat_kv_heads".to_string()));
        }

        let expected_group = n_heads / n_kv_heads;
        if group_size != expected_group {
            return Err(MetalError::InvalidShape("Invalid group size for repeat_kv_heads".to_string()));
        }

        let input = history.tensor.clone();
        let input_dims = input.dims();
        if input_dims.len() != 3 || input_dims[2] != head_dim {
            return Err(MetalError::InvalidShape("Invalid input dimensions for repeat_kv_heads".to_string()));
        }

        if input_dims[1] != history.active_seq {
            return Err(MetalError::InvalidShape("Invalid input dimensions for repeat_kv_heads".to_string()));
        }

        let canonical_heads = batch * n_kv_heads;
        let repeated_heads = batch * n_heads;

        match input_dims[0] {
            heads if heads == canonical_heads => ctx.call::<RepeatKvHeadsOp>((
                input,
                group_size as u32,
                batch as u32,
                n_kv_heads as u32,
                n_heads as u32,
                history.active_seq as u32,
                head_dim as u32,
                history.cache_capacity as u32,
            )),
            heads if heads == repeated_heads => {
                // The KV cache already stores repeated heads in [batch * n_heads, seq, head_dim].
                Ok(input)
            }
            _ => Err(MetalError::InvalidShape("Invalid input dimensions for repeat_kv_heads".to_string())),
        }
    }

    fn gather_cache_history(cache: &Tensor<T>, steps: usize, ctx: &mut Context<T>) -> Result<CacheHistory<T>, MetalError> {
        let (view, cache_capacity) = ctx.kv_cache_history_view(cache, steps)?;
        Ok(CacheHistory {
            tensor: view,
            active_seq: steps,
            cache_capacity,
        })
    }
}

#[derive(Clone)]
struct CacheHistory<T: TensorElement> {
    tensor: Tensor<T>,
    active_seq: usize,
    cache_capacity: usize,
}

impl<T: TensorElement> CacheHistory<T> {
    fn from_tensor(tensor: Tensor<T>) -> Result<Self, MetalError> {
        let dims = tensor.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape("Cache history tensors must be rank-3".to_string()));
        }
        if dims[1] == 0 {
            return Err(MetalError::InvalidShape(
                "Cache history tensors must have non-zero sequence length".to_string(),
            ));
        }

        Ok(Self {
            cache_capacity: dims[1],
            active_seq: dims[1],
            tensor,
        })
    }
}
