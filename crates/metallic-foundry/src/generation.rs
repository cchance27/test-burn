use std::{
    sync::mpsc, time::{Duration, Instant}
};

use metallic_cli_helpers::app_event::AppEvent;

use crate::{Tokenizer, error::MetalError};

/// Generation configuration (defaults chosen by user)
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Initial KV cache headroom in tokens beyond the current prompt length.
    /// This lets us avoid over-allocating the KV pool when typical generations are short.
    /// If generation exceeds this, we currently do not grow the KV cache mid-run.
    pub kv_initial_headroom_tokens: usize,
    /// Random seed for sampling. If None, a random seed will be generated.
    pub seed: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
            kv_initial_headroom_tokens: 256,
            seed: None,
        }
    }
}

/// High-level end-to-end generation pipeline with token streaming support (Foundry backend)
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming(
    foundry: &mut crate::Foundry,
    model: &crate::model::CompiledModel,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    let tokenizer = model.tokenizer()?;
    let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
    generate_streaming_from_tokens(foundry, model, &tokenizer, &prompt_tokens, cfg, tx)
}

/// Streaming generation for the Foundry backend using pre-tokenized prompt ids.
///
/// Using pre-tokenized ids avoids re-tokenizing/formatting in the hot path and keeps perf metrics consistent.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_from_tokens(
    foundry: &mut crate::Foundry,
    model: &crate::model::CompiledModel,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    let generation_start = Instant::now();
    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);

    let mut decode_scratch = Vec::new();
    let mut decoded_chunk = String::new();

    let callback = |token_id: u32, prefill_duration: Duration| -> Result<bool, MetalError> {
        if let Some(text) = tokenizer.decode_token_arc(token_id, &mut decoded_chunk, &mut decode_scratch)?
            && tx
                .send(AppEvent::Token {
                    text,
                    prompt_processing: prefill_duration,
                    iteration: None, // Foundry doesn't report per-token time yet in callback
                })
                .is_err()
        {
            return Ok(false);
        }
        Ok(true)
    };

    model.generate_with_seed_streaming(
        foundry,
        prompt_tokens,
        cfg.max_tokens,
        &[eos],
        cfg.temperature,
        cfg.top_k as u32,
        cfg.top_p,
        cfg.seed.unwrap_or_else(rand::random),
        callback,
    )?;

    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });
    Ok(())
}
