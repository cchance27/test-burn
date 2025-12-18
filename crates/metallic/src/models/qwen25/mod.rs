use std::{
    collections::BTreeMap, time::{Duration, Instant}
};

use metallic_instrumentation::{MetricEvent, record_metric_async};
use objc2_metal::{MTLBlitCommandEncoder as _, MTLDevice as _};

use crate::{
    Context, MetalError, Tensor, TensorElement, context::{MatmulAlphaBeta, QkvWeights, RepeatKvWorkspaceKind}, kernels::{
        backend_registry::KernelBackendKind, embedding_lookup::EmbeddingLookupOp, kv_rearrange::KvRearrangeOp, repeat_kv_heads::RepeatKvHeadsOp, rmsnorm::RMSNormOp, rope::RoPEOp, swiglu::{SwiGLUFusedActivationOp, SwiGLUOp}
    }, tensor::{QuantizedTensor, TensorType}
};

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
    /// Optional packed Q8_0 weight for the final output projection (logits).
    pub output_weight_q8: Option<crate::tensor::QuantizedQ8_0Tensor>,
    pub final_norm_gamma: Tensor<T>,
    pub rope_cos_cache: Tensor<T>,
    pub rope_sin_cache: Tensor<T>,
}

impl<T: TensorElement> Qwen25<T> {
    #[inline]
    fn output_weight_q8_transpose_b(logical_dims: &[usize], d_model: usize, vocab_size: usize) -> Option<bool> {
        if logical_dims.len() != 2 {
            return None;
        }
        let a = logical_dims[0];
        let b = logical_dims[1];
        if a == d_model && b == vocab_size {
            // Stored as [K, N] (d_model, vocab) already.
            Some(false)
        } else if a == vocab_size && b == d_model {
            // Stored as [N, K] (vocab, d_model) -> transpose to [K, N].
            Some(true)
        } else {
            None
        }
    }

    fn project_dense_slice(
        ctx: &mut Context<T>,
        x_flat: &Tensor<T>,
        weight: &Tensor<T>,
        range: std::ops::Range<usize>,
        bias: Tensor<T>,
    ) -> Result<Tensor<T>, MetalError> {
        let slice = weight.slice_last_dim(range)?;
        ctx.matmul(x_flat, &TensorType::Dense(&slice), false, false, Some(&bias), None, None)
    }

    fn project_quant(
        ctx: &mut Context<T>,
        x_flat: &Tensor<T>,
        tensor: &crate::tensor::QuantizedQ8_0Tensor,
        bias: Tensor<T>,
    ) -> Result<Tensor<T>, MetalError> {
        ctx.matmul(
            x_flat,
            &TensorType::Quant(QuantizedTensor::Q8_0(tensor)),
            false,
            false,
            Some(&bias),
            None,
            None,
        )
    }

    /// Embed tokens into d_model dimensional vectors
    pub fn embed(&self, tokens: &[u32], ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
        let batch = 1; // For now, assume batch size of 1
        let seq = tokens.len();

        // Build a small Shared indices tensor so host writes do not force a blit flush.
        let byte_len = std::mem::size_of_val(tokens);
        let buf = ctx
            .device
            .newBufferWithLength_options(byte_len, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;
        let mut indices = Tensor::<crate::tensor::dtypes::U32>::from_existing_buffer(
            buf,
            vec![seq],
            crate::tensor::dtypes::U32::DTYPE,
            &ctx.device,
            &ctx.command_queue,
            0,
            true,
        )?;
        let ids = indices.as_mut_slice();
        for (i, &tok) in tokens.iter().enumerate() {
            ids[i] = tok;
        }

        // Call GPU embedding lookup to produce [batch, seq, d_model] directly on device.
        let out = ctx.call::<EmbeddingLookupOp>((&self.embed_weight, &indices), None)?;

        // Ensure expected shape
        debug_assert_eq!(out.dims(), &[batch, seq, self.config.d_model]);
        Ok(out)
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
        let logits_flat = if let Some(q8) = self.output_weight_q8.as_ref()
            && T::DTYPE == crate::tensor::Dtype::F16
            && let Some(transpose_b) = Self::output_weight_q8_transpose_b(&q8.logical_dims, d_model, self.config.vocab_size)
        {
            ctx.matmul(
                &flat_hidden,
                &TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                false,
                transpose_b,
                None,
                None,
                None,
            )?
        } else {
            ctx.matmul(&flat_hidden, &TensorType::Dense(&self.output_weight), false, true, None, None, None)?
        };

        // Synchronize to ensure matmul is complete before reading values

        // Reshape back to [batch, seq, vocab_size]
        let logits = logits_flat.reshape(vec![batch, seq, self.config.vocab_size])?;

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
            output_weight_q8: None,
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

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let resid_attn = x.clone();

            // RMSNorm before Attention
            let x_normed_attn = ctx.call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32), None)?;

            // QKV GEMMs
            let m = batch * seq;
            let kv_dim = block.kv_dim;
            let x_flat = x_normed_attn.reshape(vec![m, d_model])?;
            let (q_mat, k_mat, v_mat) = ctx.qkv(
                &x_flat,
                QkvWeights::Dense {
                    fused_weight: &block.attn_qkv_weight,
                    fused_bias: &block.attn_qkv_bias,
                    d_model,
                    kv_dim,
                },
            )?;

            // Defer RoPE until after head rearrangement
            let (q_after, k_after) = (q_mat.clone(), k_mat.clone());
            // KV Head Rearrangement
            let n_heads = self.config.n_heads;
            let n_kv_heads = self.config.n_kv_heads;
            let head_dim = d_model / n_heads;
            let kv_head_dim = kv_dim / n_kv_heads;

            let q_heads = ctx.call::<KvRearrangeOp>(
                (
                    q_after,
                    d_model as u32,
                    head_dim as u32,
                    n_heads as u32,
                    n_heads as u32,
                    head_dim as u32,
                    seq as u32,
                ),
                None,
            )?;
            let k_heads = ctx.call::<KvRearrangeOp>(
                (
                    k_after,
                    kv_dim as u32,
                    kv_head_dim as u32,
                    n_kv_heads as u32,
                    n_kv_heads as u32,
                    kv_head_dim as u32,
                    seq as u32,
                ),
                None,
            )?;
            let v_heads = ctx.call::<KvRearrangeOp>(
                (
                    v_mat,
                    kv_dim as u32,
                    kv_head_dim as u32,
                    n_kv_heads as u32,
                    n_kv_heads as u32,
                    kv_head_dim as u32,
                    seq as u32,
                ),
                None,
            )?;

            // Apply RoPE per head on Q and K using head_dim (and kv_head_dim)
            let q_heads_after_rope = ctx.call::<RoPEOp>(
                (
                    q_heads,
                    self.rope_cos_cache.clone(),
                    self.rope_sin_cache.clone(),
                    head_dim as u32,
                    seq as u32,
                    0,
                ),
                None,
            )?;
            let k_heads_after_rope = ctx.call::<RoPEOp>(
                (
                    k_heads,
                    self.rope_cos_cache.clone(),
                    self.rope_sin_cache.clone(),
                    kv_head_dim as u32,
                    seq as u32,
                    0,
                ),
                None,
            )?;

            // Repeat K and V to match Q head count for SDPA (GQA)
            let group_size = n_heads / n_kv_heads;

            let k_history = CacheHistory::from_tensor(k_heads_after_rope)?;
            let v_history = CacheHistory::from_tensor(v_heads)?;

            let k_repeated = Qwen25::repeat_kv_heads(
                &k_history,
                group_size,
                batch,
                n_kv_heads,
                n_heads,
                kv_head_dim,
                layer_idx,
                RepeatKvWorkspaceKind::Key,
                ctx,
            )?;

            let v_repeated = Qwen25::repeat_kv_heads(
                &v_history,
                group_size,
                batch,
                n_kv_heads,
                n_heads,
                kv_head_dim,
                layer_idx,
                RepeatKvWorkspaceKind::Value,
                ctx,
            )?;

            // SDPA (causal mask enabled)
            let attn_out_heads = ctx.scaled_dot_product_attention(&q_heads_after_rope, &k_repeated, &v_repeated, true)?;

            // Attention Output Reassembly
            let attn_out_reshaped = attn_out_heads
                .reshape(vec![batch, n_heads, seq, head_dim])?
                .permute(&[0, 2, 1, 3], ctx)?
                .reshape(vec![batch, seq, d_model])?;

            // Attention output projection: prefer quant when available
            let attn_out_flat = {
                let a = &attn_out_reshaped.reshape(vec![m, d_model])?;
                if let Some(q8) = &block.attn_out_weight_q8 {
                    ctx.matmul(a, &TensorType::Quant(QuantizedTensor::Q8_0(q8)), false, true, None, None, None)?
                } else {
                    ctx.matmul(a, &TensorType::Dense(&block.attn_out_weight), false, true, None, None, None)?
                }
            };
            let attn_out = attn_out_flat.reshape(vec![batch, seq, d_model])?;

            // Residual Add
            x = resid_attn.add_elem(&attn_out, ctx)?;

            // --- MLP Block ---
            let resid_mlp = x.clone();

            // RMSNorm before MLP
            let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32), None)?;
            let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model])?;

            // FFN using extracted SwiGLU; prefer quant paths if available
            let ffn_output_flat = if block.ffn_gate_q8.is_some() && block.ffn_up_q8.is_some() && block.ffn_down_q8.is_some() {
                let gate_q8 = block.ffn_gate_q8.as_ref().unwrap();
                let up_q8 = block.ffn_up_q8.as_ref().unwrap();
                let down_q8 = block.ffn_down_q8.as_ref().unwrap();
                // gate and up projections (quant + bias)
                let gate = ctx.matmul(
                    &x_normed_mlp_flat,
                    &TensorType::Quant(QuantizedTensor::Q8_0(gate_q8)),
                    false,
                    false,
                    Some(&block.ffn_gate_bias),
                    None,
                    None,
                )?;
                let up = ctx.matmul(
                    &x_normed_mlp_flat,
                    &TensorType::Quant(QuantizedTensor::Q8_0(up_q8)),
                    false,
                    false,
                    Some(&block.ffn_up_bias),
                    None,
                    None,
                )?;
                // fused activation on the intermediate [m, ff_dim]
                let gate_leading = if gate.dims().len() >= 2 {
                    gate.dims()[gate.dims().len() - 2] as u32
                } else {
                    gate.dims().last().copied().unwrap_or(1) as u32
                };
                let up_leading = if up.dims().len() >= 2 {
                    up.dims()[up.dims().len() - 2] as u32
                } else {
                    up.dims().last().copied().unwrap_or(1) as u32
                };
                let hidden = ctx.call::<SwiGLUFusedActivationOp>(
                    (
                        gate,
                        block.ffn_gate_bias.clone(),
                        up,
                        block.ffn_up_bias.clone(),
                        gate_leading,
                        up_leading,
                    ),
                    None,
                )?;
                // down projection (quant) with fused bias add
                ctx.matmul(
                    &hidden,
                    &TensorType::Quant(QuantizedTensor::Q8_0(down_q8)),
                    false,
                    false,
                    Some(&block.ffn_down_bias),
                    None,
                    None,
                )?
            } else {
                ctx.call::<SwiGLUOp>(
                    (
                        &x_normed_mlp_flat,
                        &block.ffn_gate,
                        &block.ffn_gate_bias,
                        &block.ffn_up,
                        &block.ffn_up_bias,
                        &block.ffn_down,
                        &block.ffn_down_bias,
                        Some(&block.ffn_gate_up_weight),
                    ),
                    None,
                )?
            };
            let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;

            // Residual Add
            x = resid_mlp.add_elem(&ffn_output, ctx)?;
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32), None)?;

        Ok(final_normed)
    }

    #[allow(clippy::type_complexity)]
    /// Step-forward for autoregressive generation with KV caching.
    pub fn forward_step(
        &self,
        input: &Tensor<T>,
        pos: usize,
        ctx: &mut Context<T>,
    ) -> Result<(Tensor<T>, BTreeMap<usize, (String, BTreeMap<String, u64>)>), MetalError> {
        // TODO: we really should move tuple structs to proper zero size types
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
        let mut forward_pass_breakdown = BTreeMap::new();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_start = Instant::now();
            let mut breakdown = BTreeMap::new();
            let bytes_per_element = T::DTYPE.size_bytes();

            x = ctx.with_gpu_scope(format!("block_{}", layer_idx), |ctx| -> Result<Tensor<T>, MetalError> {
                // Accumulate CPU time between GPU calls in this block
                let mut cpu_accum = Duration::ZERO;
                let mut cpu_chk = Instant::now();
                ctx.set_pending_gpu_scope(format!("attn_residual_clone_block_{}_op", layer_idx));
                let resid_attn = x.clone();

                // RMSNorm before Attention
                ctx.set_pending_gpu_scope(format!("attn_norm_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x_normed_attn = ctx.call::<RMSNormOp>((x, block.attn_norm_gamma.clone(), d_model as u32), None)?;
                breakdown.insert("attn_norm".to_string(), (x_normed_attn.len() * bytes_per_element) as u64);
                cpu_chk = Instant::now();

                let m = batch * seq; // m is always 1 for a single token
                let kv_dim = block.kv_dim;
                let x_flat = x_normed_attn.reshape(vec![m, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                let q_bias = block.attn_qkv_bias.slice(0..d_model)?;
                let k_bias = block.attn_qkv_bias.slice(d_model..(d_model + kv_dim))?;
                let v_bias = block.attn_qkv_bias.slice((d_model + kv_dim)..(d_model + 2 * kv_dim))?;

                let quant_available =
                    block.attn_q_weight_q8.is_some() || block.attn_k_weight_q8.is_some() || block.attn_v_weight_q8.is_some();
                // Allow disabling quant path via env for troubleshooting
                let disable_quant = std::env::var("METALLIC_DISABLE_MLX_Q8").is_ok() || std::env::var("Q8_DISABLE").is_ok();

                // Prefer quant projections when available; fallback to dense slice or fused path otherwise.
                let (q_mat, k_mat, v_mat) = if quant_available && !disable_quant {
                    if let (Some(q8), Some(k8), Some(v8)) = (&block.attn_q_weight_q8, &block.attn_k_weight_q8, &block.attn_v_weight_q8) {
                        // Try fused Q8 path when shapes are compatible; otherwise fall back to per-projection
                        match ctx.qkv(
                            &x_flat,
                            QkvWeights::Quantized {
                                wq: q8,
                                wk: k8,
                                wv: v8,
                                q_bias: &q_bias,
                                k_bias: &k_bias,
                                v_bias: &v_bias,
                            },
                        ) {
                            Ok((q, k, v)) => (q, k, v),
                            Err(_) => {
                                // Fall back to mixing quant/dense projections
                                let q_mat = Self::project_quant(ctx, &x_flat, q8, q_bias.clone())?;
                                let k_mat = Self::project_quant(ctx, &x_flat, k8, k_bias.clone())?;
                                let v_mat = Self::project_quant(ctx, &x_flat, v8, v_bias.clone())?;
                                (q_mat, k_mat, v_mat)
                            }
                        }
                    } else {
                        let q_mat = match &block.attn_q_weight_q8 {
                            Some(q8) => Self::project_quant(ctx, &x_flat, q8, q_bias.clone())?,
                            None => Self::project_dense_slice(ctx, &x_flat, &block.attn_qkv_weight, 0..d_model, q_bias.clone())?,
                        };

                        let k_mat = match &block.attn_k_weight_q8 {
                            Some(k8) => Self::project_quant(ctx, &x_flat, k8, k_bias.clone())?,
                            None => Self::project_dense_slice(
                                ctx,
                                &x_flat,
                                &block.attn_qkv_weight,
                                d_model..(d_model + kv_dim),
                                k_bias.clone(),
                            )?,
                        };

                        let v_mat = match &block.attn_v_weight_q8 {
                            Some(v8) => Self::project_quant(ctx, &x_flat, v8, v_bias.clone())?,
                            None => Self::project_dense_slice(
                                ctx,
                                &x_flat,
                                &block.attn_qkv_weight,
                                (d_model + kv_dim)..(d_model + 2 * kv_dim),
                                v_bias.clone(),
                            )?,
                        };

                        (q_mat, k_mat, v_mat)
                    }
                } else {
                    ctx.qkv(
                        &x_flat,
                        QkvWeights::Dense {
                            fused_weight: &block.attn_qkv_weight,
                            fused_bias: &block.attn_qkv_bias,
                            d_model,
                            kv_dim,
                        },
                    )?
                };
                breakdown.insert(
                    "attn_qkv_proj".to_string(),
                    (q_mat.len() * bytes_per_element + k_mat.len() * bytes_per_element + v_mat.len() * bytes_per_element) as u64,
                );
                cpu_chk = Instant::now();

                // KV Head Rearrangement
                let n_heads = self.config.n_heads;
                let n_kv_heads = self.config.n_kv_heads;
                let head_dim = d_model / n_heads;
                let kv_head_dim = kv_dim / n_kv_heads;

                cpu_accum += cpu_chk.elapsed();
                let (q_heads, k_heads, v_heads) = ctx.with_gpu_scope(format!("attn_rearrange_block_{}_op", layer_idx), |ctx| {
                    let q_heads = ctx.call::<KvRearrangeOp>(
                        (
                            q_mat,
                            d_model as u32,
                            head_dim as u32,
                            n_heads as u32,
                            n_heads as u32,
                            head_dim as u32,
                            seq as u32,
                        ),
                        None,
                    )?;
                    let k_heads = ctx.call::<KvRearrangeOp>(
                        (
                            k_mat,
                            kv_dim as u32,
                            kv_head_dim as u32,
                            n_kv_heads as u32,
                            n_kv_heads as u32,
                            kv_head_dim as u32,
                            seq as u32,
                        ),
                        None,
                    )?;
                    let v_heads = ctx.call::<KvRearrangeOp>(
                        (
                            v_mat,
                            kv_dim as u32,
                            kv_head_dim as u32,
                            n_kv_heads as u32,
                            n_kv_heads as u32,
                            kv_head_dim as u32,
                            seq as u32,
                        ),
                        None,
                    )?;
                    Ok::<_, MetalError>((q_heads, k_heads, v_heads))
                })?;
                breakdown.insert(
                    "attn_rearrange".to_string(),
                    (q_heads.len() * bytes_per_element + k_heads.len() * bytes_per_element + v_heads.len() * bytes_per_element) as u64,
                );
                cpu_chk = Instant::now();

                // Apply RoPE using the pre-computed cache for the current position
                let position_offset = pos as u32;
                cpu_accum += cpu_chk.elapsed();
                let (q_heads_after_rope, k_heads_after_rope) = ctx.with_gpu_scope(format!("rope_block_{}_op", layer_idx), |ctx| {
                    let q_heads_after_rope = ctx.call::<RoPEOp>(
                        (
                            q_heads,
                            self.rope_cos_cache.clone(),
                            self.rope_sin_cache.clone(),
                            head_dim as u32,
                            seq as u32,
                            position_offset,
                        ),
                        None,
                    )?;
                    let k_heads_after_rope = ctx.call::<RoPEOp>(
                        (
                            k_heads,
                            self.rope_cos_cache.clone(),
                            self.rope_sin_cache.clone(),
                            kv_head_dim as u32,
                            seq as u32,
                            position_offset,
                        ),
                        None,
                    )?;
                    Ok::<_, MetalError>((q_heads_after_rope, k_heads_after_rope))
                })?;
                breakdown.insert(
                    "rope".to_string(),
                    (q_heads_after_rope.len() * bytes_per_element + k_heads_after_rope.len() * bytes_per_element) as u64,
                );
                cpu_chk = Instant::now();

                // Update the KV cache with the new K and V values
                let group_size = n_heads / n_kv_heads;
                cpu_accum += cpu_chk.elapsed();
                ctx.with_gpu_scope(format!("kv_cache_block_{}_op", layer_idx), |ctx| {
                    ctx.write_kv_step(layer_idx, pos, group_size, &k_heads_after_rope, &v_heads)
                })?;
                breakdown.insert(
                    "kv_cache".to_string(),
                    (k_heads_after_rope.len() * bytes_per_element + v_heads.len() * bytes_per_element) as u64,
                );
                cpu_chk = Instant::now();

                // Create a view over the repeated KV cache for attention
                let cache_entry = ctx
                    .kv_caches()
                    .get(&layer_idx)
                    .cloned()
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not found", layer_idx)))?;
                let k_repeated_history = Qwen25::gather_cache_history(&cache_entry.k, pos + 1, ctx)?;
                let v_repeated_history = Qwen25::gather_cache_history(&cache_entry.v, pos + 1, ctx)?;
                cpu_accum += cpu_chk.elapsed();
                let (k_repeated, v_repeated) = ctx.with_gpu_scope(format!("kv_repeat_block_{}_op", layer_idx), |ctx| {
                    let k_repeated = Qwen25::repeat_kv_heads(
                        &k_repeated_history,
                        group_size,
                        batch,
                        n_kv_heads,
                        n_heads,
                        kv_head_dim,
                        layer_idx,
                        RepeatKvWorkspaceKind::Key,
                        ctx,
                    )?;
                    let v_repeated = Qwen25::repeat_kv_heads(
                        &v_repeated_history,
                        group_size,
                        batch,
                        n_kv_heads,
                        n_heads,
                        kv_head_dim,
                        layer_idx,
                        RepeatKvWorkspaceKind::Value,
                        ctx,
                    )?;
                    Ok::<_, MetalError>((k_repeated, v_repeated))
                })?;
                breakdown.insert(
                    "kv_repeat".to_string(),
                    (k_repeated.len() * bytes_per_element + v_repeated.len() * bytes_per_element) as u64,
                );
                cpu_chk = Instant::now();

                // SDPA (causal mask enabled)
                cpu_accum += cpu_chk.elapsed();
                let attn_out_heads = ctx.with_gpu_scope(format!("sdpa_block_{}_op", layer_idx), |ctx| {
                    ctx.scaled_dot_product_attention_with_offset(&q_heads_after_rope, &k_repeated, &v_repeated, true, pos)
                })?;
                breakdown.insert("sdpa".to_string(), (attn_out_heads.len() * bytes_per_element) as u64);
                cpu_chk = Instant::now();

                // Attention Output Reassembly
                ctx.set_pending_gpu_scope(format!("attn_reassembly_block_{}_op", layer_idx));
                let attn_out_reshaped_1 = attn_out_heads.reshape(vec![batch, n_heads, seq, head_dim])?;
                let attn_out_permuted = attn_out_reshaped_1.permute(&[0, 2, 1, 3], ctx)?;
                let attn_out_reshaped = attn_out_permuted.reshape(vec![batch, seq, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                let attn_out = ctx
                    .with_gpu_scope(format!("attn_output_block_{}_op", layer_idx), |ctx| {
                        let a = &attn_out_reshaped.reshape(vec![m, d_model])?;
                        if let Some(q8) = &block.attn_out_weight_q8 {
                            // Fuse residual: y = A*x + residual
                            ctx.call::<crate::kernels::matmul_gemv::MatmulGemvAddmmOp>(
                                (
                                    a,
                                    TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                    None,
                                    Some(&resid_attn.reshape(vec![m, d_model])?),
                                    1.0,
                                    1.0,
                                ),
                                None,
                            )
                        } else {
                            // Dense fallback: use backend-heuristic addmm (MPS/MLX) with out=residual
                            let out = resid_attn.reshape(vec![m, d_model])?;
                            ctx.matmul(
                                a,
                                &TensorType::Dense(&block.attn_out_weight),
                                false,
                                true,
                                None,
                                Some(MatmulAlphaBeta {
                                    output: &out,
                                    alpha: 1.0,
                                    beta: 1.0,
                                }),
                                None,
                            )
                        }
                    })?
                    .reshape(vec![batch, seq, d_model])?;
                breakdown.insert("attn_output".to_string(), (attn_out.len() * bytes_per_element) as u64);
                cpu_chk = Instant::now();

                // Residual already fused
                let x = attn_out;

                // --- MLP Block ---
                ctx.set_pending_gpu_scope(format!("mlp_residual_clone_block_{}_op", layer_idx));
                let resid_mlp = x.clone();
                ctx.set_pending_gpu_scope(format!("mlp_norm_block_{}_op", layer_idx));
                cpu_accum += cpu_chk.elapsed();
                let x_normed_mlp = ctx.call::<RMSNormOp>((x, block.ffn_norm_gamma.clone(), d_model as u32), None)?;
                cpu_chk = Instant::now();
                let x_normed_mlp_flat = x_normed_mlp.reshape(vec![m, d_model])?;

                cpu_accum += cpu_chk.elapsed();
                // FFN with per-projection Q8 fallbacks:
                // - gate/up: use quant matmul when available, else dense matmul (no bias here)
                // - fused activation adds biases
                // - down: use quant matmul when available, else dense matmul; then add bias
                let ffn_output_flat = ctx.with_gpu_scope(format!("mlp_swiglu_block_{}_op", layer_idx), |ctx| {
                    // gate/up projections fused (quant) -> [m, ff_dim] each
                    // gate/up projections fused (quant) -> [m, ff_dim] each
                    let hidden = if block.ffn_gate_q8.is_some() && block.ffn_up_q8.is_some() {
                        let gq8 = block.ffn_gate_q8.as_ref().unwrap();
                        let uq8 = block.ffn_up_q8.as_ref().unwrap();
                        ctx.set_pending_gpu_scope(format!("mlp_swiglu_fused_block_{}_op", layer_idx));
                        ctx.call::<crate::kernels::matmul_gemv::MatmulGemvQ8SwiGluOp>(
                            (
                                &x_normed_mlp_flat,
                                (&QuantizedTensor::Q8_0(gq8), &QuantizedTensor::Q8_0(uq8)),
                                (Some(&block.ffn_gate_bias), Some(&block.ffn_up_bias)),
                            ),
                            None,
                        )?
                    } else {
                        // Fallback to separate matmuls when either weight is dense
                        ctx.set_pending_gpu_scope(format!("mlp_gate_proj_block_{}_op", layer_idx));
                        let gate_lin = {
                            if let Some(gq8) = &block.ffn_gate_q8 {
                                ctx.matmul(
                                    &x_normed_mlp_flat,
                                    &TensorType::Quant(QuantizedTensor::Q8_0(gq8)),
                                    false,
                                    false,
                                    None,
                                    None,
                                    None,
                                )?
                            } else {
                                let gd = block.ffn_gate.dims();
                                let gate_t = if gd.len() == 2 && gd[0] == d_model { false } else { true };
                                ctx.matmul(
                                    &x_normed_mlp_flat,
                                    &TensorType::Dense(&block.ffn_gate),
                                    false,
                                    gate_t,
                                    None,
                                    None,
                                    None,
                                )?
                            }
                        };
                        ctx.set_pending_gpu_scope(format!("mlp_up_proj_block_{}_op", layer_idx));
                        let up_lin = {
                            if let Some(uq8) = &block.ffn_up_q8 {
                                ctx.matmul(
                                    &x_normed_mlp_flat,
                                    &TensorType::Quant(QuantizedTensor::Q8_0(uq8)),
                                    false,
                                    false,
                                    None,
                                    None,
                                    None,
                                )?
                            } else {
                                let ud = block.ffn_up.dims();
                                let up_t = if ud.len() == 2 && ud[0] == d_model { false } else { true };
                                ctx.matmul(&x_normed_mlp_flat, &TensorType::Dense(&block.ffn_up), false, up_t, None, None, None)?
                            }
                        };

                        // fused activation on [m, ff_dim]
                        let gate_leading = if gate_lin.dims().len() >= 2 {
                            gate_lin.dims()[gate_lin.dims().len() - 2] as u32
                        } else {
                            gate_lin.dims().last().copied().unwrap_or(1) as u32
                        };
                        let up_leading = if up_lin.dims().len() >= 2 {
                            up_lin.dims()[up_lin.dims().len() - 2] as u32
                        } else {
                            up_lin.dims().last().copied().unwrap_or(1) as u32
                        };
                        ctx.call::<SwiGLUFusedActivationOp>(
                            (
                                gate_lin,
                                block.ffn_gate_bias.clone(),
                                up_lin,
                                block.ffn_up_bias.clone(),
                                gate_leading,
                                up_leading,
                            ),
                            None,
                        )?
                    };

                    // down projection -> [m, d_model]
                    ctx.set_pending_gpu_scope(format!("mlp_down_proj_block_{}_op", layer_idx));
                    if let Some(dq8) = &block.ffn_down_q8 {
                        // Quant down projection with fused bias and residual.
                        let out = ctx.call::<crate::kernels::matmul_gemv::MatmulGemvAddmmOp>(
                            (
                                &hidden,
                                TensorType::Quant(QuantizedTensor::Q8_0(dq8)),
                                Some(&block.ffn_down_bias),
                                Some(&resid_mlp.reshape(vec![m, d_model])?),
                                1.0,
                                1.0,
                            ),
                            None,
                        )?;
                        Ok(out)
                    } else {
                        // Dense fallback with fused bias+residual using backend-heuristic addmm
                        let dd = block.ffn_down.dims();
                        let hidden_cols = hidden.dims()[1];
                        let down_t = if dd.len() == 2 && dd[0] == hidden_cols { false } else { true };
                        let out = resid_mlp.reshape(vec![m, d_model])?;
                        ctx.matmul(
                            &hidden,
                            &TensorType::Dense(&block.ffn_down),
                            false,
                            down_t,
                            None,
                            Some(MatmulAlphaBeta {
                                output: &out,
                                alpha: 1.0,
                                beta: 1.0,
                            }),
                            None,
                        )
                    }
                })?;

                breakdown.insert("mlp_swiglu".to_string(), (ffn_output_flat.len() * bytes_per_element) as u64);
                ctx.set_pending_gpu_scope(format!("mlp_reshape_block_{}_op", layer_idx));
                let ffn_output = ffn_output_flat.reshape(vec![batch, seq, d_model])?;

                // Residual already fused
                let x = ffn_output;
                breakdown.insert("mlp_output".to_string(), (x.len() * bytes_per_element) as u64);

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

            forward_pass_breakdown.insert(layer_idx, (format!("Block {}", layer_idx + 1), breakdown));
        }

        // Final RMSNorm after all blocks
        let final_normed = ctx.call::<RMSNormOp>((x, self.final_norm_gamma.clone(), self.config.d_model as u32), None)?;

        Ok((final_normed, forward_pass_breakdown))
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
        layer_idx: usize,
        workspace_kind: RepeatKvWorkspaceKind,
        ctx: &mut Context<T>,
    ) -> Result<Tensor<T>, MetalError> {
        let prefer_shared = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy).backend == KernelBackendKind::Graph;
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
            heads if heads == canonical_heads => ctx.call::<RepeatKvHeadsOp>(
                (
                    input,
                    group_size as u32,
                    batch as u32,
                    n_kv_heads as u32,
                    n_heads as u32,
                    history.active_seq as u32,
                    head_dim as u32,
                    history.cache_capacity as u32,
                    layer_idx as u32,
                    workspace_kind,
                    prefer_shared,
                ),
                None,
            ),
            heads if heads == repeated_heads => {
                // Check which backend is currently selected
                let current_backend = ctx.backend_registry().select_sdpa(KernelBackendKind::Legacy).backend;

                // For MPSGraph backend, we can work with zero-copy tensor views directly
                // even if they have non-zero offsets or strided layouts, but we need to be careful
                // to avoid triggering MLIR pass assertions, so we only optimize when not prefer_shared
                if current_backend == KernelBackendKind::Graph && !prefer_shared {
                    // Use the input tensor directly with its existing layout
                    // The MPSGraph binding logic can handle non-zero offsets via MPSNDArray views
                    Ok(input)
                } else if prefer_shared {
                    // For legacy backend or when shared memory is preferred, proceed with workspace copy
                    let mut workspace = ctx.acquire_repeat_kv_workspace(
                        layer_idx,
                        workspace_kind,
                        repeated_heads,
                        history.active_seq,
                        history.cache_capacity,
                        head_dim,
                        prefer_shared,
                    )?;

                    ctx.prepare_tensors_for_active_cmd(&[&input, &workspace])?;
                    let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
                    let encoder = command_buffer.get_blit_encoder()?;

                    let head_dim_bytes = head_dim * input.dtype.size_bytes();
                    let copy_bytes = history.active_seq * head_dim_bytes;
                    unsafe {
                        for head in 0..repeated_heads {
                            let src_offset = input.offset + head * history.cache_capacity * head_dim_bytes;
                            let dst_offset = workspace.offset + head * history.active_seq * head_dim_bytes;
                            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &input.buf,
                                src_offset,
                                &workspace.buf,
                                dst_offset,
                                copy_bytes,
                            );
                        }
                    }
                    ctx.mark_tensor_pending(&workspace);
                    workspace.strides = Tensor::<T>::compute_strides(workspace.dims());
                    Ok(workspace)
                } else {
                    // The KV cache already stores repeated heads in [batch * n_heads, seq, head_dim].
                    Ok(input)
                }
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

        // For newly created tensors (like from rearrange), capacity and active sequence are the same
        // For KV cache history views, this will be properly set by the gather_cache_history function
        Ok(Self {
            cache_capacity: dims[1],
            active_seq: dims[1],
            tensor,
        })
    }
}
