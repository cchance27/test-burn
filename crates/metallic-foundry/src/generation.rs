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

/// High-level end-to-end generation pipeline with token streaming support (Foundry backend)
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming(
    foundry: &mut crate::Foundry,
    model: Arc<crate::model::CompiledModel>,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    let tokenizer = model.tokenizer()?;
    let debug_tokenize = std::env::var("METALLIC_DEBUG_TOKENIZE").is_ok();
    let disable_chat_template = std::env::var("METALLIC_DISABLE_CHAT_TEMPLATE").is_ok();
    let prompt_tokens = if debug_tokenize {
        let formatted = if disable_chat_template {
            prompt.to_string()
        } else {
            tokenizer.format_single_turn_chat_prompt(prompt)?
        };
        let toks = tokenizer.encode(&formatted)?;
        let head_n = 64usize.min(toks.len());
        let decoded_head = tokenizer
            .decode_lossless(&toks[..head_n])
            .unwrap_or_else(|_| "<decode_error>".to_string());
        let max_chars = 800usize;
        let shown = formatted.chars().take(max_chars).collect::<String>();
        let suffix = if formatted.chars().count() > max_chars {
            "…(truncated)"
        } else {
            ""
        };
        eprintln!(
            "[metallic][debug] encode_single_turn_chat_prompt disable_chat_template={} chars={} tokens={} head_ids={:?}\n[metallic][debug] decoded_head:\n{}\n[metallic][debug] formatted_prompt_head:\n{}{}",
            disable_chat_template,
            formatted.chars().count(),
            toks.len(),
            &toks[..head_n],
            decoded_head,
            shown,
            suffix
        );
        toks
    } else if disable_chat_template {
        tokenizer.encode(prompt)?
    } else {
        tokenizer.encode_single_turn_chat_prompt(prompt)?
    };
    generate_streaming_from_tokens(foundry, model, &tokenizer, &prompt_tokens, cfg, tx)
}

fn system_prompt_from_env() -> Option<Arc<str>> {
    std::env::var("METALLIC_SYSTEM_PROMPT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(Arc::<str>::from)
}

fn build_single_turn_messages_value(_tokenizer: &BPETokenizer, prompt: &str) -> Value {
    let mut sys = FxHashMap::default();
    sys.insert("role".to_string(), Value::Text("system".into()));
    // IMPORTANT: keep the default system prompt generic; if the GGUF provides a chat template,
    // it is the source of truth for formatting. Model-specific hardcoded system prompts can
    // cause dramatic behavioral shifts (e.g., refusals).
    let system = system_prompt_from_env().unwrap_or_else(|| Arc::<str>::from("You are a helpful assistant."));
    sys.insert("content".to_string(), Value::Text(system));

    let mut user = FxHashMap::default();
    user.insert("role".to_string(), Value::Text("user".into()));
    user.insert("content".to_string(), Value::Text(prompt.into()));

    Value::Array(vec![Value::Map(sys), Value::Map(user)])
}

/// Streaming generation for the Foundry backend using pre-tokenized prompt ids.
///
/// Using pre-tokenized ids avoids re-tokenizing/formatting in the hot path and keeps perf metrics consistent.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_from_tokens(
    foundry: &mut crate::Foundry,
    model: Arc<CompiledModel>,
    tokenizer: &BPETokenizer,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    static WORKFLOW_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workflows/text_generation.json"));
    static WORKFLOW: OnceLock<WorkflowSpec> = OnceLock::new();

    let workflow = WORKFLOW.get_or_init(|| serde_json::from_str(WORKFLOW_JSON).expect("invalid text_generation workflow JSON"));

    let mut models: FxHashMap<String, Arc<CompiledModel>> = FxHashMap::default();
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
    models: &FxHashMap<String, Arc<CompiledModel>>,
    tokenizer: &BPETokenizer,
    prompt_tokens: &[u32],
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
    workflow: WorkflowSpec,
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

    let mut runner = WorkflowRunner::new(foundry, models.clone());
    let wf_cfg = WorkflowRunnerConfig { workflow };

    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    // Token-driven workflows (like text_generation.json) expect pre-tokenized prompt ids.
    if wf_cfg.workflow.inputs.iter().any(|i| i.name == "prompt_tokens") {
        inputs.insert("prompt_tokens".to_string(), Value::TokensU32(prompt_tokens.to_vec().into()));
    } else {
        return Err(MetalError::InvalidOperation(
            "Workflow does not declare 'prompt_tokens' input; use generate_streaming_with_workflow_from_prompt()".into(),
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

    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
    inputs.insert("eos_token".to_string(), Value::U32(eos));
    inputs.insert("seed".to_string(), Value::U32(cfg.seed.unwrap_or_else(rand::random)));

    let _outputs = runner.run_streaming(&wf_cfg.workflow, inputs, &mut callback)?;

    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });
    Ok(())
}

/// Streaming generation for workflows that tokenize/format prompts inside the workflow graph.
///
/// This supports workflows like `workflows/multiturn_chat.json` that take `messages` input and run
/// `format_chat` + `tokenize` inside the workflow.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_with_workflow_from_prompt(
    foundry: &mut crate::Foundry,
    models: &FxHashMap<String, Arc<CompiledModel>>,
    tokenizer: &BPETokenizer,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
    workflow: WorkflowSpec,
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

    let mut runner = WorkflowRunner::new(foundry, models.clone());
    let wf_cfg = WorkflowRunnerConfig { workflow };

    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    if wf_cfg.workflow.inputs.iter().any(|i| i.name == "messages") {
        inputs.insert("messages".to_string(), build_single_turn_messages_value(tokenizer, prompt));
    } else if wf_cfg.workflow.inputs.iter().any(|i| i.name == "prompt") {
        inputs.insert("prompt".to_string(), Value::Text(prompt.into()));
    } else {
        return Err(MetalError::InvalidOperation(
            "Workflow does not declare 'messages' or 'prompt' input; expected a prompt-driven workflow".into(),
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

    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
    inputs.insert("eos_token".to_string(), Value::U32(eos));
    inputs.insert("seed".to_string(), Value::U32(cfg.seed.unwrap_or_else(rand::random)));

    let _outputs = runner.run_streaming(&wf_cfg.workflow, inputs, &mut callback)?;

    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });
    Ok(())
}
