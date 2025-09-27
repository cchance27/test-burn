use super::{Context, MetalError, Tensor};
use crate::app_event::AppEvent;
use crate::metallic::models::qwen25::Qwen25;
use crate::metallic::Tokenizer;
use app_memory_usage_fetcher::get_memory_usage_mbytes;
use rand::prelude::*;
use std::sync::mpsc;
use std::time::Instant;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Generation configuration (defaults chosen by user)
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
        }
    }
}

/// Sample from logits using top-k and top-p (nucleus) sampling.
/// - `logits` is a slice of f32 representing vocabulary logits.
///   Returns selected token index.
pub fn sample_top_k_top_p(logits: &[f32], top_k: usize, top_p: f32, temperature: f32) -> usize {
    // Apply temperature scaling and convert to positive scores
    let mut scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    // Stabilize by subtracting max before exponentiation to prevent overflow
    // Filter out any infinity/nan values first
    let finite_scaled: Vec<f32> = scaled.iter().cloned().filter(|x| x.is_finite()).collect();
    if finite_scaled.is_empty() {
        return 0; // fallback if all logits are non-finite
    }

    let m = finite_scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Apply the shift and compute exponentials
    for x in &mut scaled {
        if x.is_finite() {
            *x = (*x - m).exp();
            // Clamp extremely large values to prevent overflow
            if *x > 1e10 {
                *x = 1e10;
            }
            // Clamp extremely small values to prevent underflow
            if *x < 1e-10 {
                *x = 0.0;
            }
        } else {
            *x = 0.0; // Replace non-finite values with 0
        }
    }

    // Normalize to probabilities
    let sum: f32 = scaled.iter().sum();
    if sum <= 0.0 || sum.is_infinite() || sum.is_nan() {
        return 0usize; // fallback
    }
    for x in &mut scaled {
        *x /= sum;
    }

    // Sort indices by probability descending
    let mut idxs: Vec<usize> = (0..scaled.len()).collect();
    idxs.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top-k filtering first
    let k_cutoff = std::cmp::min(top_k, idxs.len());
    let idxs = &idxs[0..k_cutoff];

    // Then apply top-p filtering
    let mut cum = 0.0f32;
    let mut cutoff = 0usize;
    for (i, &id) in idxs.iter().enumerate() {
        cum += scaled[id];
        cutoff = i;
        if cum >= top_p || cum.is_infinite() || cum.is_nan() {
            break;
        }
    }

    let shortlist = &idxs[0..=cutoff];
    let mut shortlist_probs: Vec<f32> = shortlist.iter().map(|&i| scaled[i]).collect();
    let ssum: f32 = shortlist_probs.iter().sum();
    if ssum <= 0.0 || ssum.is_infinite() || ssum.is_nan() {
        return shortlist[0];
    }
    for p in &mut shortlist_probs {
        *p /= ssum;
    }

    // Sample using RNG (use simple rng.next_u32() -> float to avoid trait issues)
    let mut rng = rand::rng();
    let r = (rng.next_u32() as f32) / (u32::MAX as f32);
    let mut acc = 0.0f32;
    for (i, &p) in shortlist_probs.iter().enumerate() {
        acc += p;
        if r <= acc || acc.is_infinite() || acc.is_nan() {
            return shortlist[i];
        }
    }
    shortlist[shortlist.len() - 1]
}

/// High-level end-to-end generation pipeline that combines tokenization, embedding,
/// model inference, and sampling into a complete inference loop.
pub fn generate(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    prompt: &str,
    cfg: &GenerationConfig,
) -> Result<String, MetalError> {
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );
    let input_ids = tokenizer.encode(&full_prompt)?;
    let output_tokens = generate_autoregressive_with_kv_cache(qwen, tokenizer, ctx, &input_ids, cfg)?;
    let output_text = tokenizer.decode(&output_tokens)?;
    Ok(output_text)
}

/// High-level end-to-end generation pipeline with token streaming support
pub fn generate_streaming(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
    start_time: Instant,
) -> Result<(), MetalError> {
    // Build full prompt string following Qwen2.5 chat template
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    let mut token_count = 0;
    let mut token_callback = |_token_id, decoded_token| -> Result<bool, MetalError> {
        token_count += 1;
        let elapsed = start_time.elapsed();
        let tokens_per_second = token_count as f64 / elapsed.as_secs_f64();

        if tx
            .send(AppEvent::Token(decoded_token, tokens_per_second))
            .is_err()
        {
            return Ok(false); // Stop generation if UI thread has disconnected
        }
        Ok(true)
    };

    // Generate tokens using the new KV cache approach
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, &input_ids, cfg, &mut token_callback, tx)?;

    Ok(())
}

/// High-level autoregressive generation loop using Qwen25 with KV caches for debugging.
/// This implementation processes the full context each time for comparison.
pub fn generate_autoregressive_with_kv_cache(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    input_ids: &[u32],
    cfg: &GenerationConfig,
) -> Result<Vec<u32>, MetalError> {
    let mut result = Vec::new();
    let mut callback = |token_id, _decoded_token| -> Result<bool, MetalError> {
        result.push(token_id);
        Ok(true)
    };

    let (tx, _) = mpsc::channel();
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, input_ids, cfg, &mut callback, &tx)?;

    Ok(result)
}

/// High-level autoregressive generation loop with streaming support using Qwen25 with KV Caching.
pub fn generate_autoregressive_with_kv_cache_streaming<F>(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    input_ids: &[u32],
    cfg: &GenerationConfig,
    token_callback: &mut F,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError>
where
    F: FnMut(u32, String) -> Result<bool, MetalError>,
{
    // Pre-allocate KV cache for all layers
    let n_layers = qwen.config.n_layers;
    let seq_len = qwen.config.seq_len;
    let n_kv_heads = qwen.config.n_kv_heads;
    let d_model = qwen.config.d_model;
    let n_heads = qwen.config.n_heads;
    let kv_dim = d_model * n_kv_heads / n_heads;
    let kv_head_dim = kv_dim / n_kv_heads;
    let batch_size = 1; // Assuming batch size of 1 for now

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, seq_len, batch_size * n_kv_heads, kv_head_dim)?;
    }

    // --- Prompt Processing Pass ---
    // Process the prompt token by token to warm up the KV cache.
    let mut logits_tensor: Option<Tensor> = None;
    if !input_ids.is_empty() {
        ctx.clear_cache(); // It's okay to clear the resource cache
        for (i, &token_id) in input_ids.iter().enumerate() {
            let input_tensor = qwen.embed(&[token_id], ctx)?;
            let hidden_states = qwen.forward_step(&input_tensor, i, ctx)?;
            logits_tensor = Some(qwen.output(&hidden_states, ctx)?);
        }
    }

    let mut generated_ids = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let vocab_size = qwen.config.vocab_size;
    let mut next_token;

    if let Some(logits_tensor) = logits_tensor {
        // Extract logits for the very last token of the prompt
        let logits = logits_tensor.to_vec();
        let vocab_logits = logits[0..vocab_size].to_vec();

        // Sample the first token
        next_token = sample_top_k_top_p(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;

        println!("[KV] Prompt processing produced next_token: {}", next_token);
    } else {
        // If there's no prompt, start with token 0.
        next_token = 0;
    }

    generated_ids.push(next_token);
    let decoded_token = tokenizer.decode_lossless(&[next_token])?;
    if !token_callback(next_token, decoded_token)? {
        return Ok(());
    }

    // --- Autoregressive Generation Loop ---
    // Now, generate tokens one by one using the KV cache.
    for i in 0..cfg.max_tokens - 1 {
        ctx.reset_pool();
        ctx.clear_cache();

        // Embed just the single last token
        let input_tensor = qwen.embed(&[next_token], ctx)?;
        
        // Run a single forward step
        let current_pos = prompt_len + i;
        let hidden_states = qwen.forward_step(&input_tensor, current_pos, ctx)?;
        let logits_tensor = qwen.output(&hidden_states, ctx)?;

        if let Some(stats) = ctx.get_cache_stats() {
            let app_mem = get_memory_usage_mbytes();
            let memory_usage = format!(
                "App: {:.2} GB | Pool: {:.2} / {:.2} MB | Cache: G{}/D{}/S{}",
                app_mem.unwrap(),
                ctx.pool.pooled_bytes_allocated as f32 / 1024.0 / 1024.0,
                ctx.pool.total_capacity() as f32 / 1024.0 / 1024.0,
                stats.gemm_cache_size,
                stats.descriptor_cache_size,
                stats.sdpa_cache_size
            );
            if tx.send(AppEvent::MemoryUpdate(memory_usage)).is_err() {
                return Ok(()); // Stop if UI thread is gone
            }
        }

        let logits = logits_tensor.to_vec();
        
        // Since seq_len is 1, the logits are just the first (and only) vocab block
        let vocab_logits = logits[0..vocab_size].to_vec();

        // Sample the next token
        next_token = sample_top_k_top_p(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;

        println!("[KV]  Step {}: token={}, logits={:?}", i, next_token, &vocab_logits[..10]);

        generated_ids.push(next_token);

        // Callback and check for EOS
        let decoded_token = tokenizer.decode_lossless(&[next_token])?;
        if !token_callback(next_token, decoded_token)? {
            break;
        }
        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        if next_token == eos_token_id {
            break;
        }
    }

    Ok(())
}
