use super::{Context, MetalError, Tensor};
use crate::metallic::Tokenizer;
use crate::metallic::models::Qwen25;
use rand::prelude::*;

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
    let mut output_tokens = Vec::new();
    let callback = |token_id, _decoded_token| -> Result<bool, MetalError> {
        output_tokens.push(token_id);
        // Continue generation
        Ok(true)
    };

    // Use the streaming version with a callback that collects tokens
    generate_streaming(qwen, tokenizer, ctx, prompt, cfg, callback)?;

    // Decode all collected tokens
    let output_text = tokenizer.decode(&output_tokens)?;
    Ok(output_text)
}

/// High-level end-to-end generation pipeline with token streaming support
pub fn generate_streaming<F>(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    prompt: &str,
    cfg: &GenerationConfig,
    mut token_callback: F,
) -> Result<(), MetalError>
where
    F: FnMut(u32, String) -> Result<bool, MetalError>,
{
    // Build full prompt string following Qwen2.5 chat template
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    // Generate tokens using the non-KV cache approach for debugging
    generate_autoregressive_without_kv_cache_streaming(qwen, tokenizer, ctx, &input_ids, cfg, &mut token_callback)?;

    Ok(())
}

/// High-level autoregressive generation loop using Qwen25 without KV caches for debugging.
/// This implementation processes the full context each time for comparison.
pub fn generate_autoregressive_without_kv_cache(
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

    generate_autoregressive_without_kv_cache_streaming(qwen, tokenizer, ctx, input_ids, cfg, &mut callback)?;

    Ok(result)
}

/// High-level autoregressive generation loop with streaming support using Qwen25 without KV caches.
pub fn generate_autoregressive_without_kv_cache_streaming<F>(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    input_ids: &[u32],
    cfg: &GenerationConfig,
    token_callback: &mut F,
) -> Result<(), MetalError>
where
    F: FnMut(u32, String) -> Result<bool, MetalError>,
{
    // Start with input tokens
    let mut generated = input_ids.to_vec();

    // Autoregressive generation loop
    let max_gen_len = cfg.max_tokens + input_ids.len();

    while generated.len() < max_gen_len {
        // Process the entire sequence so far
        let current_ids = generated.clone();

        // Embed all tokens so far
        let input_tensor = qwen.embed(&current_ids, ctx)?;

        // Run through the full model
        let hidden_states = qwen.forward(&input_tensor, ctx)?;

        // Apply output projection
        let logits_tensor = qwen.output(&hidden_states, ctx)?;

        // Extract logits for the last token
        let logits_dims = logits_tensor.dims();
        let logits = logits_tensor.to_vec();

        // Convert logits to vocab-size slice; assume logits shape [batch, seq, vocab]
        let vocab_size = qwen.config.vocab_size;
        let vocab_logits = if logits_dims.len() >= 3 && logits_dims[logits_dims.len() - 1] == vocab_size {
            // Properly shaped logits [batch, seq, vocab]
            let seq_len = logits_dims[logits_dims.len() - 2];
            let start_idx = (seq_len - 1) * vocab_size; // Get the last sequence position
            let end_idx = start_idx + vocab_size;
            if end_idx <= logits.len() {
                logits[start_idx..end_idx].to_vec()
            } else {
                // Fallback: pad with zeros
                let mut padded = vec![0.0; vocab_size];
                let copy_len = std::cmp::min(end_idx - start_idx, logits.len());
                padded[..copy_len].copy_from_slice(&logits[..copy_len]);
                padded
            }
        } else if logits.len() >= vocab_size {
            // Fallback to original method - but get the last vocab_size elements, not the first
            let seq_len = logits.len() / vocab_size;
            let start_idx = (seq_len - 1) * vocab_size; // Get the last sequence position
            let end_idx = start_idx + vocab_size;
            if end_idx <= logits.len() {
                logits[start_idx..end_idx].to_vec()
            } else {
                // If we can't determine the correct slice, fall back to first vocab_size elements
                logits[..vocab_size].to_vec()
            }
        } else {
            // If we don't have enough logits, pad with zeros
            let mut padded = logits;
            padded.resize(vocab_size, 0.0);
            padded
        };

        // Sample next token
        let next_token = sample_top_k_top_p(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;

        generated.push(next_token);

        // Call the callback with the new token and its decoded form
        let decoded_token = tokenizer.decode_lossless(&[next_token])?;
        if !token_callback(next_token, decoded_token)? {
            // If callback returns false, stop generation
            break;
        }

        // Check for EOS token
        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645); // Qwen2.5 default EOS
        if next_token == eos_token_id {
            break;
        }
    }

    Ok(())
}
