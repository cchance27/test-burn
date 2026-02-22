use std::{
    panic::{self, AssertUnwindSafe}, path::PathBuf, sync::{Arc, mpsc}, thread, time::{Duration, Instant}
};

use anyhow::Result;
use metallic_cli_helpers::prelude::*;
use metallic_foundry::{
    model::{CompiledModel, ModelBuilder}, workflow::WorkflowRunner
};
use metallic_instrumentation::{MetricEvent, record_metric_async};
use metallic_loader::ModelLoader;
use rustc_hash::FxHashMap;

use crate::{
    app::{
        events::{emit_startup_memory_update, panic_payload_message}, workflow_inputs::{
            apply_workflow_cli_inputs, build_workflow_cli_inputs, env_is_set, summarize_workflow_value, workflow_template_kwargs
        }
    }, cli
};

pub(crate) struct GenerationWorkerParams {
    pub tx: mpsc::Sender<AppEvent>,
    pub cmd_rx: mpsc::Receiver<AppEvent>,
    pub gguf_path: String,
    pub prompts: Vec<String>,
    pub worker_generation: cli::config::GenerationConfig,
    pub worker_output_format: cli::config::OutputFormat,
    pub workflow_path: Option<String>,
    pub worker_workflow_kwargs: Vec<(String, String)>,
    pub worker_thinking_override: Option<bool>,
}

pub(crate) fn spawn_generation_worker(params: GenerationWorkerParams) -> thread::JoinHandle<Result<()>> {
    let GenerationWorkerParams {
        tx,
        cmd_rx,
        gguf_path,
        prompts,
        worker_generation,
        worker_output_format,
        workflow_path,
        worker_workflow_kwargs,
        worker_thinking_override,
    } = params;

    let worker_tx = tx.clone();
    thread::spawn(move || -> Result<()> {
        let worker = || -> Result<()> {
            let workflow_cli_inputs = build_workflow_cli_inputs(&worker_workflow_kwargs, worker_thinking_override)?;
            let tokenizer_template_kwargs = workflow_template_kwargs(&workflow_cli_inputs)?;
            if !workflow_cli_inputs.is_empty() {
                tracing::info!(
                    kwarg_count = workflow_cli_inputs.len(),
                    "Loaded workflow CLI kwargs (--kwarg / --thinking)"
                );
                for (key, value) in &workflow_cli_inputs {
                    tracing::debug!(
                        key = key.as_str(),
                        value_type = value.type_name(),
                        value = summarize_workflow_value(value),
                        "Workflow CLI kwarg"
                    );
                }
            }
            if let Some(value) = workflow_cli_inputs.get("enable_thinking") {
                if let Some(enabled) = value.as_boolish() {
                    tracing::info!(enable_thinking = enabled, "Thinking override provided by CLI");
                } else {
                    tracing::debug!(
                        value_type = value.type_name(),
                        value = summarize_workflow_value(value),
                        "enable_thinking provided but not coercible to bool"
                    );
                }
            }
            if let Some(kwargs) = tokenizer_template_kwargs.as_ref() {
                tracing::debug!(
                    template_kwargs = kwargs.len(),
                    "Tokenizer/template kwargs prepared for chat rendering"
                );
            }
            // Send initial empty memory update - simplified for new system
            emit_startup_memory_update(&worker_tx)?;
            worker_tx.send(AppEvent::StatusUpdate("Loading GGUF Metadata...".to_string()))?;
            let load_start = Instant::now();
            // Report GGUF file MMAP usage
            let gguf_file_size = std::fs::metadata(&gguf_path)?.len();
            record_metric_async!(MetricEvent::GgufFileMmap {
                size_bytes: gguf_file_size
            });
            emit_startup_memory_update(&worker_tx)?;
            // Foundry engine (the only engine now)
            let workflow_override: Option<metallic_foundry::workflow::WorkflowSpec> = if let Some(path) = &workflow_path {
                let f = std::fs::File::open(path)?;
                Some(serde_json::from_reader(f)?)
            } else {
                None
            };
            let has_workflow_model_resources = workflow_override
                .as_ref()
                .and_then(|w| w.resources.as_ref())
                .is_some_and(|r| !r.models.is_empty());
            let mut single_model_loaded: Option<Box<dyn metallic_loader::LoadedModel>> = None;
            let routed_spec_path: Option<PathBuf> = if has_workflow_model_resources {
                None
            } else {
                worker_tx.send(AppEvent::StatusUpdate("Detecting architecture...".to_string()))?;
                let model_loaded = ModelLoader::from_file(&gguf_path)?;
                let routing = metallic_foundry::model_routing::resolve_model_routing_from_loaded_model(model_loaded.as_ref())
                    .map_err(anyhow::Error::msg)?;
                worker_tx.send(AppEvent::StatusUpdate(format!(
                    "Detected architecture: {} (rule: {})",
                    routing.architecture, routing.matched_rule
                )))?;
                single_model_loaded = Some(model_loaded);
                Some(routing.spec_path)
            };
            worker_tx.send(AppEvent::StatusUpdate("Initializing Foundry...".to_string()))?;
            let mut foundry = metallic_foundry::Foundry::new()?;
            worker_tx.send(AppEvent::StatusUpdate("Building compiled model(s)...".to_string()))?;
            let mut models_owned: FxHashMap<String, Arc<CompiledModel>> = FxHashMap::default();
            if has_workflow_model_resources {
                let resources = workflow_override
                    .as_ref()
                    .and_then(|w| w.resources.as_ref())
                    .expect("has_workflow_model_resources implies resources");
                for m in &resources.models {
                    let model_loaded = ModelLoader::from_file(&m.gguf_path)?;
                    let model = ModelBuilder::new()
                        .with_spec_file(PathBuf::from(&m.spec_path))?
                        .with_model(model_loaded)
                        .build(&mut foundry)?;
                    models_owned.insert(m.id.clone(), Arc::new(model));
                }
            } else {
                let spec_path = routed_spec_path.expect("routed_spec_path required for single-model Foundry");
                let model_loaded = single_model_loaded
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("single-model Foundry expected a preloaded model"))?;
                let model = ModelBuilder::new()
                    .with_spec_file(spec_path)?
                    .with_model(model_loaded)
                    .build(&mut foundry)?;
                let model_id = workflow_override
                    .as_ref()
                    .and_then(|w| w.default_model.clone())
                    .unwrap_or_else(|| "llm".to_string());
                models_owned.insert(model_id, Arc::new(model));
            }
            // Report memory metrics for Foundry model
            for model in models_owned.values() {
                model.report_memory_metrics();
            }
            emit_startup_memory_update(&worker_tx)?;
            let load_duration = load_start.elapsed();
            worker_tx.send(AppEvent::ModelLoadComplete(load_duration))?;
            worker_tx.send(AppEvent::StatusUpdate("Initializing tokenizer...".to_string()))?;
            let tokenization_start = Instant::now();
            let tokenizer_model_id = workflow_override
                .as_ref()
                .and_then(|w| w.default_model.as_deref())
                .or_else(|| models_owned.keys().next().map(|s| s.as_str()))
                .unwrap_or("llm");
            let tokenizer_model = models_owned
                .get(tokenizer_model_id)
                .ok_or_else(|| anyhow::anyhow!("Workflow default_model '{tokenizer_model_id}' not found"))?;
            let tokenizer = tokenizer_model.tokenizer()?;
            let interactive = matches!(worker_output_format, cli::config::OutputFormat::Tui);
            let workflow_wants_messages = workflow_override
                .as_ref()
                .is_some_and(|wf| wf.inputs.iter().any(|i| i.name == "messages"));
            // Prompt-driven workflows (e.g. `inputs: ["messages", ...]`) are supported in TUI mode.
            // For multi-turn, we maintain message history in the TUI loop and rely on Foundry's KV cache
            // reuse + prefill incremental suffix handling to avoid replaying the full prompt every turn.
            let debug_tokenize = env_is_set("METALLIC_DEBUG_TOKENIZE");
            let disable_chat_template = env_is_set("METALLIC_DISABLE_CHAT_TEMPLATE");
            let first_prompt = prompts.first().map(|s| s.trim()).unwrap_or("");
            let cfg = metallic_foundry::generation::GenerationConfig {
                max_tokens: worker_generation.max_tokens,
                temperature: worker_generation.temperature as f32,
                top_p: worker_generation.top_p as f32,
                min_p: worker_generation.min_p as f32,
                top_k: worker_generation.top_k,
                repeat_penalty: worker_generation.repeat_penalty as f32,
                repeat_last_n: worker_generation.repeat_last_n,
                presence_penalty: worker_generation.presence_penalty as f32,
                frequency_penalty: worker_generation.frequency_penalty as f32,
                kv_initial_headroom_tokens: 0,
                seed: worker_generation.seed,
            };
            // Penalties are applied on-GPU in the workflow sampler and are compatible with batching.
            // `models_owned` is already FxHashMap<String, Arc<CompiledModel>> and generation is workflow-only.
            let models = models_owned; // Ownership transfer / move for clarity, though we could just use models_owned.
            if !interactive {
                worker_tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
                if debug_tokenize {
                    let formatted = if disable_chat_template {
                        prompts[0].clone()
                    } else {
                        tokenizer.format_single_turn_chat_prompt_with_kwargs(&prompts[0], tokenizer_template_kwargs.as_ref())?
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
                        "[metallic][debug] main: disable_chat_template={} chars={} tokens={} head_ids={:?}\n[metallic][debug] decoded_head:\n{}\n[metallic][debug] formatted_prompt_head:\n{}{}",
                        disable_chat_template,
                        formatted.chars().count(),
                        toks.len(),
                        &toks[..head_n],
                        decoded_head,
                        shown,
                        suffix
                    );
                    if env_is_set("METALLIC_DEBUG_TOKENIZE_FULL") {
                        eprintln!("[metallic][debug] token_ids_full={:?}", toks);
                    }
                }
                let tokens0 = if disable_chat_template {
                    tokenizer.encode(&prompts[0])?
                } else {
                    tokenizer.encode_single_turn_chat_prompt_with_kwargs(&prompts[0], tokenizer_template_kwargs.as_ref())?
                };
                let tokenization_duration = tokenization_start.elapsed();
                worker_tx.send(AppEvent::TokenizationComplete(tokenization_duration))?;
                worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
                if workflow_wants_messages {
                    if prompts.first().map(|s| s.trim()).unwrap_or("").is_empty() {
                        return Err(anyhow::anyhow!(
                            "Workflow declares 'messages' input (prompt-driven workflow) but no prompt was provided. Pass a prompt argument, or use `--output-format tui` for interactive chat."
                        ));
                    }
                    use metallic_foundry::workflow::Value as WfValue;
                    fn sys_prompt() -> Option<String> {
                        std::env::var("METALLIC_SYSTEM_PROMPT")
                            .ok()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                    }
                    fn msg(role: &str, content: &str) -> WfValue {
                        let mut map = FxHashMap::default();
                        map.insert("role".to_string(), WfValue::Text(role.into()));
                        map.insert("content".to_string(), WfValue::Text(content.to_string().into()));
                        WfValue::Map(map)
                    }
                    let workflow = workflow_override.as_ref().expect("workflow_override present");
                    let mut runner = WorkflowRunner::new(models);
                    let system = sys_prompt();
                    for (turn_idx, turn_prompt) in prompts.iter().enumerate() {
                        if turn_idx > 0 {
                            worker_tx.send(AppEvent::UserPrompt(turn_prompt.to_string()))?;
                        }
                        let messages_input: Vec<WfValue> = if turn_idx == 0 {
                            if let Some(system) = system.as_deref() {
                                vec![msg("system", system), msg("user", turn_prompt)]
                            } else {
                                vec![msg("user", turn_prompt)]
                            }
                        } else {
                            vec![msg("user", turn_prompt)]
                        };
                        // Best-effort token metrics for throughput/UI (actual tokenization is done inside the workflow).
                        let mut metrics_tokens: Vec<u32> = if turn_idx == 0 {
                            tokens0.clone()
                        } else if disable_chat_template {
                            tokenizer.encode(turn_prompt)?
                        } else {
                            tokenizer.encode_chat_continuation_prompt_with_kwargs(turn_prompt, tokenizer_template_kwargs.as_ref())?
                        };
                        // Remove BOS if present to avoid polluting the context in the middle of a chat.
                        if turn_idx > 0
                            && let Some(bos) = tokenizer.special_tokens().bos_token_id
                            && !metrics_tokens.is_empty()
                            && metrics_tokens[0] == bos
                        {
                            metrics_tokens.remove(0);
                        }
                        worker_tx.send(AppEvent::TokenCount(metrics_tokens.len()))?;
                        worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
                        let mut inputs: FxHashMap<String, WfValue> = FxHashMap::default();
                        inputs.insert("messages".to_string(), WfValue::Array(messages_input));
                        inputs.insert("max_tokens".to_string(), WfValue::U32(cfg.max_tokens as u32));
                        inputs.insert("temperature".to_string(), WfValue::F32(cfg.temperature));
                        inputs.insert("top_k".to_string(), WfValue::U32(cfg.top_k as u32));
                        inputs.insert("top_p".to_string(), WfValue::F32(cfg.top_p));
                        inputs.insert("min_p".to_string(), WfValue::F32(cfg.min_p));
                        inputs.insert("repeat_penalty".to_string(), WfValue::F32(cfg.repeat_penalty));
                        inputs.insert("repeat_last_n".to_string(), WfValue::Usize(cfg.repeat_last_n));
                        inputs.insert("presence_penalty".to_string(), WfValue::F32(cfg.presence_penalty));
                        inputs.insert("frequency_penalty".to_string(), WfValue::F32(cfg.frequency_penalty));
                        inputs.insert("seed".to_string(), WfValue::U32(cfg.seed.unwrap_or(42)));
                        // Prefer runner auto-injection of `eos_token` when the workflow declares it.
                        if !workflow.inputs.iter().any(|i| i.name == "eos_token") {
                            let eos = tokenizer.special_tokens().eos_token_id.ok_or_else(|| {
                                metallic_foundry::MetalError::InvalidOperation(
                                    "Tokenizer metadata missing required 'eos_token_id'".to_string(),
                                )
                            })?;
                            inputs.insert("eos_token".to_string(), WfValue::U32(eos));
                        }
                        apply_workflow_cli_inputs(&mut inputs, &workflow_cli_inputs);
                        let mut decode_scratch = Vec::new();
                        let mut decoded_chunk = String::new();
                        let mut on_token = |token_id: u32,
                                            prefill: Duration,
                                            setup: Duration,
                                            iter: Option<Duration>|
                         -> Result<bool, metallic_foundry::MetalError> {
                            if let Some(text) = tokenizer.decode_token_arc(token_id, &mut decoded_chunk, &mut decode_scratch)?
                                && worker_tx
                                    .send(AppEvent::Token {
                                        text,
                                        setup_duration: Some(setup),
                                        prompt_processing: prefill,
                                        iteration: iter,
                                    })
                                    .is_err()
                            {
                                return Ok(false);
                            }
                            Ok(true)
                        };
                        let gen_start = Instant::now();
                        match runner.run_streaming(&mut foundry, workflow, inputs, &mut on_token) {
                            Ok(_outputs) => {
                                let _ = worker_tx.send(AppEvent::GenerationComplete {
                                    total_generation_time: gen_start.elapsed(),
                                });
                            }
                            Err(err) => {
                                alert::emit_error(&worker_tx, format!("Generation failed: {err:#}"));
                                let _ = worker_tx.send(AppEvent::GenerationComplete {
                                    total_generation_time: gen_start.elapsed(),
                                });
                                worker_tx.send(AppEvent::StatusUpdate("Waiting for input...".to_string()))?;
                            }
                        }
                    }
                    return Ok(());
                }
                for (turn_idx, turn_prompt) in prompts.iter().enumerate() {
                    let mut current_tokens = if turn_idx == 0 {
                        tokens0.clone()
                    } else if disable_chat_template {
                        tokenizer.encode(turn_prompt)?
                    } else {
                        tokenizer.encode_chat_continuation_prompt_with_kwargs(turn_prompt, tokenizer_template_kwargs.as_ref())?
                    };
                    // Remove BOS if present to avoid polluting the context in the middle of a chat.
                    if turn_idx > 0
                        && let Some(bos) = tokenizer.special_tokens().bos_token_id
                        && !current_tokens.is_empty()
                        && current_tokens[0] == bos
                    {
                        current_tokens.remove(0);
                    }
                    worker_tx.send(AppEvent::TokenCount(current_tokens.len()))?;
                    let workflow = workflow_override
                        .clone()
                        .unwrap_or_else(metallic_foundry::generation::default_text_generation_workflow);
                    let overrides = workflow_override.as_ref().map(|_| &workflow_cli_inputs);
                    metallic_foundry::generation::generate_streaming_from_tokens_with_workflow(
                        &mut foundry,
                        &models,
                        &tokenizer,
                        &current_tokens,
                        &cfg,
                        &worker_tx,
                        workflow,
                        overrides,
                    )?;
                }
            } else if workflow_wants_messages {
                // TUI multi-turn for message-driven workflows:
                // - Feed only user-turn deltas into the workflow to avoid re-prefilling assistant text.
                // - Preserve KV cache in Foundry sessions; prefill consumes only the delta tokens.
                use metallic_foundry::workflow::Value as WfValue;
                fn sys_prompt() -> Option<String> {
                    std::env::var("METALLIC_SYSTEM_PROMPT")
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                }
                fn msg(role: &str, content: &str) -> WfValue {
                    let mut map = FxHashMap::default();
                    map.insert("role".to_string(), WfValue::Text(role.into()));
                    map.insert("content".to_string(), WfValue::Text(content.to_string().into()));
                    WfValue::Map(map)
                }
                let workflow = workflow_override.as_ref().expect("workflow_override present");
                let mut runner = WorkflowRunner::new(models);
                let system = sys_prompt();
                let mut is_first_turn = true;
                // If the user provided multiple prompts on the CLI for a TUI session, treat them
                // as queued turns.
                let mut queued_cli_turns = prompts.iter().skip(1);
                if debug_tokenize && !first_prompt.is_empty() {
                    let formatted = if disable_chat_template {
                        prompts[0].clone()
                    } else {
                        tokenizer.format_single_turn_chat_prompt_with_kwargs(&prompts[0], tokenizer_template_kwargs.as_ref())?
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
                        "[metallic][debug] main: disable_chat_template={} chars={} tokens={} head_ids={:?}\n[metallic][debug] decoded_head:\n{}\n[metallic][debug] formatted_prompt_head:\n{}{}",
                        disable_chat_template,
                        formatted.chars().count(),
                        toks.len(),
                        &toks[..head_n],
                        decoded_head,
                        shown,
                        suffix
                    );
                    if env_is_set("METALLIC_DEBUG_TOKENIZE_FULL") {
                        eprintln!("[metallic][debug] token_ids_full={:?}", toks);
                    }
                }
                loop {
                    let user_prompt: String = if is_first_turn {
                        // If the TUI is launched without an initial prompt, wait for user input.
                        if first_prompt.is_empty() {
                            worker_tx.send(AppEvent::StatusUpdate("Waiting for input...".to_string()))?;
                            match cmd_rx.recv() {
                                Ok(AppEvent::Input(input)) => {
                                    worker_tx.send(AppEvent::StatusUpdate("Processing input...".to_string()))?;
                                    input
                                }
                                Ok(_) => continue,
                                Err(_) => break,
                            }
                        } else {
                            prompts[0].clone()
                        }
                    } else if let Some(turn_prompt) = queued_cli_turns.next() {
                        worker_tx.send(AppEvent::UserPrompt(turn_prompt.to_string()))?;
                        worker_tx.send(AppEvent::StatusUpdate("Processing queued prompt...".to_string()))?;
                        turn_prompt.to_string()
                    } else {
                        match cmd_rx.recv() {
                            Ok(AppEvent::Input(input)) => {
                                worker_tx.send(AppEvent::StatusUpdate("Processing input...".to_string()))?;
                                input
                            }
                            Ok(_) => continue,
                            Err(_) => break,
                        }
                    };
                    let messages_input: Vec<WfValue> = if is_first_turn {
                        if let Some(system) = system.as_deref() {
                            vec![msg("system", system), msg("user", &user_prompt)]
                        } else {
                            vec![msg("user", &user_prompt)]
                        }
                    } else {
                        vec![msg("user", &user_prompt)]
                    };
                    is_first_turn = false;
                    worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
                    let mut inputs: FxHashMap<String, WfValue> = FxHashMap::default();
                    inputs.insert("messages".to_string(), WfValue::Array(messages_input));
                    inputs.insert("max_tokens".to_string(), WfValue::U32(cfg.max_tokens as u32));
                    inputs.insert("temperature".to_string(), WfValue::F32(cfg.temperature));
                    inputs.insert("top_k".to_string(), WfValue::U32(cfg.top_k as u32));
                    inputs.insert("top_p".to_string(), WfValue::F32(cfg.top_p));
                    inputs.insert("min_p".to_string(), WfValue::F32(cfg.min_p));
                    inputs.insert("repeat_penalty".to_string(), WfValue::F32(cfg.repeat_penalty));
                    inputs.insert("repeat_last_n".to_string(), WfValue::Usize(cfg.repeat_last_n));
                    inputs.insert("presence_penalty".to_string(), WfValue::F32(cfg.presence_penalty));
                    inputs.insert("frequency_penalty".to_string(), WfValue::F32(cfg.frequency_penalty));
                    inputs.insert("seed".to_string(), WfValue::U32(cfg.seed.unwrap_or(42)));
                    // Prefer runner auto-injection of `eos_token` when the workflow declares it.
                    if !workflow.inputs.iter().any(|i| i.name == "eos_token") {
                        let eos = tokenizer.special_tokens().eos_token_id.ok_or_else(|| {
                            metallic_foundry::MetalError::InvalidOperation("Tokenizer metadata missing required 'eos_token_id'".to_string())
                        })?;
                        inputs.insert("eos_token".to_string(), WfValue::U32(eos));
                    }
                    apply_workflow_cli_inputs(&mut inputs, &workflow_cli_inputs);
                    let mut decode_scratch = Vec::new();
                    let mut decoded_chunk = String::new();
                    let mut on_token = |token_id: u32,
                                        prefill: Duration,
                                        setup: Duration,
                                        iter: Option<Duration>|
                     -> Result<bool, metallic_foundry::MetalError> {
                        if let Some(text) = tokenizer.decode_token_arc(token_id, &mut decoded_chunk, &mut decode_scratch)?
                            && worker_tx
                                .send(AppEvent::Token {
                                    text,
                                    setup_duration: Some(setup),
                                    prompt_processing: prefill,
                                    iteration: iter,
                                })
                                .is_err()
                        {
                            return Ok(false);
                        }
                        Ok(true)
                    };
                    let gen_start = Instant::now();
                    match runner.run_streaming(&mut foundry, workflow, inputs, &mut on_token) {
                        Ok(_outputs) => {
                            let _ = worker_tx.send(AppEvent::GenerationComplete {
                                total_generation_time: gen_start.elapsed(),
                            });
                        }
                        Err(err) => {
                            alert::emit_error(&worker_tx, format!("Generation failed: {err:#}"));
                            let _ = worker_tx.send(AppEvent::GenerationComplete {
                                total_generation_time: gen_start.elapsed(),
                            });
                            worker_tx.send(AppEvent::StatusUpdate("Waiting for input...".to_string()))?;
                        }
                    }
                }
            } else {
                let mut is_first_turn = true;
                let mut current_tokens: Vec<u32> = if !first_prompt.is_empty() {
                    is_first_turn = false;
                    let toks = if disable_chat_template {
                        tokenizer.encode(&prompts[0])?
                    } else {
                        tokenizer.encode_single_turn_chat_prompt_with_kwargs(&prompts[0], tokenizer_template_kwargs.as_ref())?
                    };
                    let tokenization_duration = tokenization_start.elapsed();
                    worker_tx.send(AppEvent::TokenizationComplete(tokenization_duration))?;
                    worker_tx.send(AppEvent::TokenCount(toks.len()))?;
                    toks
                } else {
                    worker_tx.send(AppEvent::StatusUpdate("Waiting for input...".to_string()))?;
                    Vec::new()
                };
                let mut queued_cli_turns = prompts.iter().skip(1);
                loop {
                    if current_tokens.is_empty() {
                        // Consume queued CLI prompts first, otherwise wait for user input.
                        if let Some(turn_prompt) = queued_cli_turns.next() {
                            worker_tx.send(AppEvent::UserPrompt(turn_prompt.to_string()))?;
                            worker_tx.send(AppEvent::StatusUpdate("Processing queued prompt...".to_string()))?;
                            let mut remove_bos = false;
                            current_tokens = if is_first_turn {
                                is_first_turn = false;
                                if disable_chat_template {
                                    tokenizer.encode(turn_prompt)?
                                } else {
                                    tokenizer.encode_single_turn_chat_prompt_with_kwargs(turn_prompt, tokenizer_template_kwargs.as_ref())?
                                }
                            } else {
                                remove_bos = true;
                                tokenizer.encode_chat_continuation_prompt_with_kwargs(turn_prompt, tokenizer_template_kwargs.as_ref())?
                            };
                            // Remove BOS if present to avoid polluting the context in middle of generation
                            if remove_bos
                                && let Some(bos) = tokenizer.special_tokens().bos_token_id
                                && !current_tokens.is_empty()
                                && current_tokens[0] == bos
                            {
                                current_tokens.remove(0);
                            }
                            worker_tx.send(AppEvent::TokenCount(current_tokens.len()))?;
                            continue;
                        }
                        match cmd_rx.recv() {
                            Ok(AppEvent::Input(input)) => {
                                worker_tx.send(AppEvent::StatusUpdate("Processing input...".to_string()))?;
                                let mut remove_bos = false;
                                current_tokens = if is_first_turn {
                                    is_first_turn = false;
                                    if disable_chat_template {
                                        tokenizer.encode(&input)?
                                    } else {
                                        tokenizer.encode_single_turn_chat_prompt_with_kwargs(&input, tokenizer_template_kwargs.as_ref())?
                                    }
                                } else {
                                    remove_bos = true;
                                    tokenizer.encode_chat_continuation_prompt_with_kwargs(&input, tokenizer_template_kwargs.as_ref())?
                                };
                                // Remove BOS if present to avoid polluting the context in middle of generation
                                if remove_bos
                                    && let Some(bos) = tokenizer.special_tokens().bos_token_id
                                    && !current_tokens.is_empty()
                                    && current_tokens[0] == bos
                                {
                                    current_tokens.remove(0);
                                }
                                worker_tx.send(AppEvent::TokenCount(current_tokens.len()))?;
                                continue;
                            }
                            Ok(_) => continue,
                            Err(_) => break,
                        }
                    }
                    let workflow = workflow_override
                        .clone()
                        .unwrap_or_else(metallic_foundry::generation::default_text_generation_workflow);
                    let overrides = workflow_override.as_ref().map(|_| &workflow_cli_inputs);
                    metallic_foundry::generation::generate_streaming_from_tokens_with_workflow(
                        &mut foundry,
                        &models,
                        &tokenizer,
                        &current_tokens,
                        &cfg,
                        &worker_tx,
                        workflow,
                        overrides,
                    )?;
                    // If the user provided multiple prompts on the CLI for a TUI session, consume them
                    // as queued user turns before switching to interactive input.
                    if let Some(turn_prompt) = queued_cli_turns.next() {
                        // Echo queued CLI prompts into the transcript so interactive users don't see the chat "lag"
                        // behind (the model will answer this queued prompt next).
                        worker_tx.send(AppEvent::UserPrompt(turn_prompt.to_string()))?;
                        worker_tx.send(AppEvent::StatusUpdate("Processing queued prompt...".to_string()))?;
                        current_tokens =
                            tokenizer.encode_chat_continuation_prompt_with_kwargs(turn_prompt, tokenizer_template_kwargs.as_ref())?;
                        // Remove BOS if present to avoid polluting the context in middle of generation
                        if let Some(bos) = tokenizer.special_tokens().bos_token_id
                            && !current_tokens.is_empty()
                            && current_tokens[0] == bos
                        {
                            current_tokens.remove(0);
                        }
                        worker_tx.send(AppEvent::TokenCount(current_tokens.len()))?;
                        continue;
                    }
                    // Wait for user input for continuous chat
                    match cmd_rx.recv() {
                        Ok(AppEvent::Input(input)) => {
                            worker_tx.send(AppEvent::StatusUpdate("Processing input...".to_string()))?;
                            current_tokens =
                                tokenizer.encode_chat_continuation_prompt_with_kwargs(&input, tokenizer_template_kwargs.as_ref())?;
                            // Remove BOS if present to avoid polluting the context in middle of generation
                            if let Some(bos) = tokenizer.special_tokens().bos_token_id
                                && !current_tokens.is_empty()
                                && current_tokens[0] == bos
                            {
                                current_tokens.remove(0);
                            }
                            worker_tx.send(AppEvent::TokenCount(current_tokens.len()))?;
                        }
                        Ok(_) => {}      // Ignore other events
                        Err(_) => break, // Channel closed (app exit)
                    }
                }
            }
            worker_tx.send(AppEvent::StatusUpdate("Done.".to_string()))?;
            Ok(())
        };
        match panic::catch_unwind(AssertUnwindSafe(worker)) {
            Ok(result) => {
                if let Err(err) = &result {
                    let message = format!("Generation failed: {err:#}");
                    alert::emit_error(&worker_tx, message);
                }
                result
            }
            Err(payload) => {
                let message = format!("Generation thread panicked: {}", panic_payload_message(payload));
                alert::emit_error(&worker_tx, message.clone());
                Err(anyhow::anyhow!(message))
            }
        }
    })
}
