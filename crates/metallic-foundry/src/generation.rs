use std::{
    sync::{Arc, OnceLock, mpsc}, time::{Duration, Instant}
};

use metallic_cli_helpers::app_event::AppEvent;
use rustc_hash::FxHashMap;

use crate::{
    BPETokenizer, error::MetalError, model::CompiledModel, workflow::{Value, WorkflowRunner, WorkflowRunnerConfig, WorkflowSpec}
};

/// Generation configuration (defaults chosen by user)
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
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
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            kv_initial_headroom_tokens: 256,
            seed: None,
        }
    }
}

/// Return the default token-driven text generation workflow bundled with Foundry.
#[must_use]
pub fn default_text_generation_workflow() -> WorkflowSpec {
    static WORKFLOW_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workflows/text_generation.json"));
    static WORKFLOW: OnceLock<WorkflowSpec> = OnceLock::new();
    WORKFLOW
        .get_or_init(|| serde_json::from_str(WORKFLOW_JSON).expect("invalid text_generation workflow JSON"))
        .clone()
}

/// Streaming generation for the Foundry backend using a caller-provided workflow + model map.
///
/// This is the entrypoint used by the CLI when running user-provided workflow JSON that may define
/// multiple models/resources.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_from_tokens_with_workflow(
    foundry: &mut crate::Foundry,
    models: &FxHashMap<String, Arc<CompiledModel>>,
    tokenizer: &BPETokenizer,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
    workflow: WorkflowSpec,
    workflow_input_overrides: Option<&FxHashMap<String, Value>>,
) -> Result<(), MetalError> {
    let generation_start = Instant::now();

    let mut decode_scratch = Vec::new();
    let mut decoded_chunk = String::new();

    let mut callback =
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

    let mut runner = WorkflowRunner::new(models.clone());
    let wf_cfg = WorkflowRunnerConfig { workflow };

    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    // Token-driven workflows (like text_generation.json) expect pre-tokenized prompt ids.
    if wf_cfg.workflow.inputs.iter().any(|i| i.name == "prompt_tokens") {
        inputs.insert("prompt_tokens".to_string(), Value::TokensU32(prompt_tokens.to_vec()));
    } else {
        return Err(MetalError::InvalidOperation(
            "Workflow does not declare required 'prompt_tokens' input".into(),
        ));
    }
    inputs.insert("max_tokens".to_string(), Value::Usize(cfg.max_tokens));
    inputs.insert("temperature".to_string(), Value::F32(cfg.temperature));
    inputs.insert("top_k".to_string(), Value::U32(cfg.top_k as u32));
    inputs.insert("top_p".to_string(), Value::F32(cfg.top_p));
    inputs.insert("min_p".to_string(), Value::F32(cfg.min_p));
    inputs.insert("repeat_penalty".to_string(), Value::F32(cfg.repeat_penalty));
    inputs.insert("repeat_last_n".to_string(), Value::Usize(cfg.repeat_last_n));
    inputs.insert("presence_penalty".to_string(), Value::F32(cfg.presence_penalty));
    inputs.insert("frequency_penalty".to_string(), Value::F32(cfg.frequency_penalty));

    // Prefer runner auto-injection of `eos_token` (from the default model's tokenizer) when the workflow declares it.
    if !wf_cfg.workflow.inputs.iter().any(|i| i.name == "eos_token") {
        let eos = tokenizer
            .special_tokens()
            .eos_token_id
            .ok_or_else(|| MetalError::InvalidOperation("Tokenizer metadata missing required 'eos_token_id'".to_string()))?;
        inputs.insert("eos_token".to_string(), Value::U32(eos));
    }
    inputs.insert("seed".to_string(), Value::U32(cfg.seed.unwrap_or_else(rand::random)));
    if let Some(overrides) = workflow_input_overrides {
        for (name, value) in overrides {
            inputs.insert(name.clone(), value.clone());
        }
    }

    let _outputs = runner.run_streaming(foundry, &wf_cfg.workflow, inputs, &mut callback)?;

    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });
    Ok(())
}
