use std::{
    sync::mpsc, time::{Duration, Instant}
};

use metallic_cli_helpers::app_event::AppEvent;
use rustc_hash::FxHashMap;

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
    static WORKFLOW_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workflows/text_generation.json"));
    static WORKFLOW: std::sync::OnceLock<crate::workflow::WorkflowSpec> = std::sync::OnceLock::new();

    let workflow = WORKFLOW.get_or_init(|| serde_json::from_str(WORKFLOW_JSON).expect("invalid text_generation workflow JSON"));

    let mut models: FxHashMap<String, &crate::model::CompiledModel> = FxHashMap::default();
    models.insert("llm".to_string(), model);

    generate_streaming_from_tokens_with_workflow(foundry, &models, tokenizer, prompt_tokens, cfg, tx, workflow.clone())
}

/// Streaming generation for the Foundry backend using a caller-provided workflow + model map.
///
/// This is the entrypoint used by the CLI when running user-provided workflow JSON that may define
/// multiple models/resources.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_from_tokens_with_workflow(
    foundry: &mut crate::Foundry,
    models: &FxHashMap<String, &crate::model::CompiledModel>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
    workflow: crate::workflow::WorkflowSpec,
) -> Result<(), MetalError> {
    let generation_start = Instant::now();
    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);

    let mut decode_scratch = Vec::new();
    let mut decoded_chunk = String::new();

    let callback =
        |token_id: u32, prefill_duration: Duration, setup_duration: Duration, iteration: Option<Duration>| -> Result<bool, MetalError> {
            if let Some(text) = tokenizer.decode_token_arc(token_id, &mut decoded_chunk, &mut decode_scratch)?
                && tx
                    .send(AppEvent::Token {
                        text,
                        setup_duration: Some(setup_duration),
                        prompt_processing: prefill_duration,
                        iteration,
                    })
                    .is_err()
            {
                return Ok(false);
            }
            Ok(true)
        };

    let mut runner = crate::workflow::WorkflowRunner::new(foundry, models.clone());
    let wf_cfg = crate::workflow::WorkflowRunnerConfig { workflow };

    let mut inputs: FxHashMap<String, crate::workflow::Value> = FxHashMap::default();
    inputs.insert(
        "prompt_tokens".to_string(),
        crate::workflow::Value::TokensU32(prompt_tokens.to_vec().into()),
    );
    inputs.insert("max_tokens".to_string(), crate::workflow::Value::Usize(cfg.max_tokens));
    inputs.insert("temperature".to_string(), crate::workflow::Value::F32(cfg.temperature));
    inputs.insert("top_k".to_string(), crate::workflow::Value::U32(cfg.top_k as u32));
    inputs.insert("top_p".to_string(), crate::workflow::Value::F32(cfg.top_p));
    inputs.insert("eos_token".to_string(), crate::workflow::Value::U32(eos));
    inputs.insert(
        "seed".to_string(),
        crate::workflow::Value::U32(cfg.seed.unwrap_or_else(rand::random)),
    );

    let _outputs = runner.run_streaming(&wf_cfg, inputs, callback)?;

    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });
    Ok(())
}
