use super::{Context, MetalError, Tensor};
use crate::metallic::Tokenizer;
use crate::metallic::models::Qwen25;
use rand::prelude::*;

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
    println!("Starting generation with prompt: {}", prompt);
    // Build full prompt string following Qwen2.5 chat template
    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";
    let full_prompt = format!(
        "{im_start}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{im_end}\n{im_start}user\n{prompt}{im_end}\n{im_start}assistant\n",
    );
    println!("Full prompt: {:?}", full_prompt);

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    // Generate tokens using the non-KV cache approach for debugging
    let generated_ids = generate_autoregressive_without_kv_cache(qwen, tokenizer, ctx, &input_ids, cfg)?;

    // Only decode the new tokens generated after the prompt, and trim at EOS if present
    let start = input_ids.len();
    let eos_id_opt = tokenizer.special_tokens().eos_token_id;
    let tail = &generated_ids[start..];
    let decode_slice: &[u32] = if let Some(eos_id) = eos_id_opt {
        if let Some(pos) = tail.iter().position(|&t| t == eos_id) {
            &tail[..pos]
        } else {
            tail
        }
    } else {
        tail
    };
    let output_text = tokenizer.decode(decode_slice)?;

    Ok(output_text)
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
    println!(
        "Starting autoregressive generation without KV cache with {} input tokens",
        input_ids.len()
    );

    // Start with input tokens
    let mut generated = input_ids.to_vec();

    // Autoregressive generation loop
    let max_gen_len = cfg.max_tokens + input_ids.len();

    println!("Starting generation loop: cur_pos={}, max_gen_len={}", input_ids.len(), max_gen_len);

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
        // Debug: print sampled token
        //println!(
        //    "Sampled token ID: {}, decoded: '{}'",
        //    next_token,
        //    tokenizer.decode(&[next_token]).unwrap_or_default()
        //);

        // Check for EOS token
        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645); // Qwen2.5 default EOS
        if next_token == eos_token_id {
            break;
        }

        // Debug output
        if generated.len() <= 20 {
            println!("Generated token {}: {}", generated.len(), next_token);
        }
    }

    Ok(generated)
}
