use std::{
    any::Any, io::{Write, stdout}, panic::{self, AssertUnwindSafe}, sync::{Arc, mpsc}, thread, time::Duration
};

use anyhow::Result;
use metallic_cli_helpers::prelude::*;
use metallic_context::{
    Context, F16Element, TensorElement, Tokenizer, gguf::{GGUFFile, model_loader::GGUFModelLoader}, kernels::{KernelBackendKind, KernelBackendOverride, KernelBackendOverrides}, profiling_state
};
use metallic_foundry::model::CompiledModel;
use metallic_instrumentation::{MetricEvent, config::AppConfig, prelude::*, record_metric_async};
use rustc_hash::FxHashMap;

mod cli;
mod tui;

use std::sync::OnceLock;

use clap::Parser;
use crossterm::event::{self, Event as CrosstermEvent, MouseButton, MouseEvent};
use ratatui::{
    Terminal, backend::{Backend, CrosstermBackend}, layout::Position
};
use tui::{App, AppResult, app::FocusArea, ui};

static LOG_SENDER: OnceLock<mpsc::Sender<AppEvent>> = OnceLock::new();

struct AppLogWriter;

impl std::io::Write for AppLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let msg = String::from_utf8_lossy(buf).trim_end().to_string();
        if let Some(tx) = LOG_SENDER.get() {
            let _ = tx.send(AppEvent::LogMessage(msg));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn main() -> AppResult<()> {
    // Ensure profiling state is initialized from the environment before anything else.
    // This avoids a race condition where the TUI might read the state before the
    // generation thread initializes it.
    profiling_state::initialize_profiling_state_from_env();

    // Parse command line arguments using CLAP
    let cli_config = cli::CliConfig::parse();

    // If output format is TUI, signal to instrumentation system to enable metrics
    if matches!(cli_config.output_format, cli::config::OutputFormat::Tui) {
        // SAFETY: This is called early in main, before other threads are spawned or environment accessed concurrently.
        unsafe {
            std::env::set_var("METALLIC_TUI_MODE", "1");
            // Note: Per-kernel profiling is controlled by:
            // 1. Runtime toggle via Ctrl+P (AppConfig::profiling_forced())
            // 2. User env var METALLIC_FOUNDRY_PER_KERNEL_PROFILING
            // We don't force it off here to allow user control.
        }
    }

    // Load instrumentation config from the environment so exporter selection honours CLI env vars.
    let app_config = AppConfig::get_or_init_from_env().map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;

    // Initialize instrumentation system with async recorder for zero-overhead metrics.
    // NOTE: The recorder always installs an in-process channel exporter (receiver returned below).
    let mut exporters: Vec<Box<dyn MetricExporter>> = Vec::new();

    if let Some(path) = app_config.metrics_jsonl_path.clone() {
        match JsonlExporter::new(&path) {
            Ok(jsonl_exporter) => exporters.push(Box::new(jsonl_exporter)),
            Err(error) => eprintln!("Failed to initialize JsonlExporter at {:?}: {}", path, error),
        }
    }

    if app_config.enable_console_metrics {
        exporters.push(Box::new(ConsoleExporter::new()));
    }

    let async_recorder = AsyncMetricRecorder::new(exporters);
    let metric_queue = async_recorder.queue.clone();

    // Initialize the global metric queue for the async macro
    metallic_instrumentation::macros::init_metric_queue(metric_queue);

    alert::init_error_logging();
    let gguf_path = cli_config.gguf_path.clone();
    let prompts = cli_config.get_prompts();
    let tui_start_processing = matches!(cli_config.output_format, cli::config::OutputFormat::Tui) && !prompts.is_empty();
    let worker_generation = cli_config.generation;
    let worker_backend = cli_config.backend;
    let worker_sdpa_backend = cli_config.sdpa_backend;
    let worker_engine = cli_config.engine;
    let worker_output_format = cli_config.output_format.clone();
    let workflow_path = cli_config.workflow.clone();

    fn env_bool(name: &str) -> bool {
        let Ok(value) = std::env::var(name) else {
            return false;
        };
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return false;
        }
        let lowered = trimmed.to_ascii_lowercase();
        !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
    }

    fn env_usize(name: &str) -> Option<usize> {
        let value = std::env::var(name).ok()?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }
        trimmed.parse::<usize>().ok()
    }

    fn env_is_set(name: &str) -> bool {
        std::env::var(name).is_ok()
    }

    // Defensive: it's easy to accidentally leave `METALLIC_IGNORE_EOS_STOP=1` in the environment from
    // benchmarking scripts. When enabled, generation can look "stuck" because workflows ignore EOS.
    if env_bool("METALLIC_IGNORE_EOS_STOP") && !matches!(worker_output_format, cli::config::OutputFormat::None) {
        tracing::warn!(
            "METALLIC_IGNORE_EOS_STOP is enabled; EOS stopping is disabled (benchmark-only setting). Unset it for normal generation."
        );
    }

    if env_bool("METALLIC_DISABLE_CHAT_TEMPLATE") {
        tracing::warn!("METALLIC_DISABLE_CHAT_TEMPLATE is enabled; prompts will be tokenized as raw text (no chat formatting).");
    }

    let (tx, rx) = mpsc::channel();
    let (cmd_tx, cmd_rx) = mpsc::channel();

    // Initialize logging based on mode
    if matches!(cli_config.output_format, cli::config::OutputFormat::Tui) {
        // In TUI mode, redirect logs to the app event channel
        let _ = LOG_SENDER.set(tx.clone());
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
            .with_writer(|| AppLogWriter)
            .with_target(false)
            .without_time() // TUI log pane is narrow, save space
            .init();
    } else {
        // In text/JSON mode, log to stderr (default) so stdout is clean for output
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
            .with_writer(std::io::stderr)
            .init();
    }

    let generation_handle = {
        let worker_tx = tx.clone();
        thread::spawn(move || -> Result<()> {
            let worker = || -> Result<()> {
                // Send initial empty memory update - simplified for new system
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Loading GGUF Metadata...".to_string()))?;
                let load_start = std::time::Instant::now();
                let gguf = GGUFFile::load_mmap_and_get_metadata(&gguf_path)?;

                // Report GGUF file MMAP usage
                let gguf_file_size = std::fs::metadata(&gguf_path)?.len();
                record_metric_async!(MetricEvent::GgufFileMmap {
                    size_bytes: gguf_file_size
                });

                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
                let mut ctx = Context::<F16Element>::new()?;

                // Apply global backend override first (affects all kernels that consult the registry)
                if let Some(choice) = worker_backend {
                    let override_policy = match choice {
                        cli::config::GlobalBackendChoice::Auto => KernelBackendOverride::Auto,
                        cli::config::GlobalBackendChoice::Legacy => KernelBackendOverride::Force(KernelBackendKind::Legacy),
                        cli::config::GlobalBackendChoice::Graph => KernelBackendOverride::Force(KernelBackendKind::Graph),
                    };
                    ctx.set_global_backend_override(override_policy);
                }

                // Then apply per-op SDPA override if provided (takes precedence for sdpa)
                if let Some(choice) = worker_sdpa_backend {
                    let override_policy = match choice {
                        cli::config::SdpaBackendChoice::Auto => KernelBackendOverride::Auto,
                        cli::config::SdpaBackendChoice::Legacy => KernelBackendOverride::Force(KernelBackendKind::Legacy),
                        cli::config::SdpaBackendChoice::Graph => KernelBackendOverride::Force(KernelBackendKind::Graph),
                    };
                    ctx.apply_backend_overrides(KernelBackendOverrides {
                        sdpa: Some(override_policy),
                    });
                }
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;

                match worker_engine {
                    cli::config::Engine::Context => {
                        worker_tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
                        let loader = GGUFModelLoader::new(gguf);
                        emit_startup_memory_update(&worker_tx)?;
                        let gguf_model = loader.load_model()?;
                        emit_startup_memory_update(&worker_tx)?;

                        worker_tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
                        let mut qwen: metallic_context::models::qwen25::Qwen25<F16Element> = gguf_model.instantiate(&mut ctx)?;

                        // Report model weights breakdown
                        report_model_weight_breakdown(&qwen);
                        emit_startup_memory_update(&worker_tx)?;

                        let load_duration = load_start.elapsed();
                        worker_tx.send(AppEvent::ModelLoadComplete(load_duration))?;

                        worker_tx.send(AppEvent::StatusUpdate("Initializing tokenizer...".to_string()))?;
                        let tokenization_start = std::time::Instant::now();
                        let tokenizer = Tokenizer::from_gguf_metadata(&gguf_model.metadata)?;
                        emit_startup_memory_update(&worker_tx)?;

                        let cfg = metallic_context::generation::GenerationConfig {
                            max_tokens: worker_generation.max_tokens,
                            temperature: worker_generation.temperature as f32,
                            top_p: worker_generation.top_p as f32,
                            top_k: worker_generation.top_k,
                            kv_initial_headroom_tokens: (worker_generation.max_tokens / 4).max(32),
                            seed: worker_generation.seed,
                        };

                        // In TUI mode, an omitted prompt means "start empty and wait for user input".
                        let prompts_for_run: Vec<String> = if prompts.is_empty() {
                            worker_tx.send(AppEvent::StatusUpdate("Waiting for input...".to_string()))?;
                            loop {
                                match cmd_rx.recv() {
                                    Ok(AppEvent::Input(input)) => break vec![input],
                                    Ok(_) => continue,
                                    Err(_) => return Ok(()),
                                }
                            }
                        } else {
                            prompts.clone()
                        };

                        worker_tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
                        let mut conversation_tokens = tokenizer.encode_single_turn_chat_prompt(&prompts_for_run[0])?;

                        let tokenization_duration = tokenization_start.elapsed();
                        worker_tx.send(AppEvent::TokenizationComplete(tokenization_duration))?;

                        emit_startup_memory_update(&worker_tx)?;

                        worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;

                        if prompts_for_run.len() == 1 {
                            worker_tx.send(AppEvent::TokenCount(conversation_tokens.len()))?;
                            metallic_context::generation::generate_streaming_from_tokens(
                                &mut qwen,
                                &tokenizer,
                                &mut ctx,
                                &conversation_tokens,
                                &cfg,
                                &worker_tx,
                            )?;
                        } else {
                            for (turn_idx, turn_prompt) in prompts_for_run.iter().enumerate() {
                                if turn_idx > 0 {
                                    let mut next_tokens = tokenizer.encode_chat_continuation_prompt(turn_prompt)?;
                                    if let Some(bos) = tokenizer.special_tokens().bos_token_id
                                        && !next_tokens.is_empty()
                                        && next_tokens[0] == bos
                                    {
                                        next_tokens.remove(0);
                                    }
                                    conversation_tokens.extend(next_tokens);
                                }

                                worker_tx.send(AppEvent::TokenCount(conversation_tokens.len()))?;
                                let generated_ids = metallic_context::generation::generate_streaming_from_tokens_collect(
                                    &mut qwen,
                                    &tokenizer,
                                    &mut ctx,
                                    &conversation_tokens,
                                    &cfg,
                                    &worker_tx,
                                )?;
                                conversation_tokens.extend(generated_ids);
                            }
                        }
                    }

                    cli::config::Engine::Foundry => {
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

                        let spec_path: Option<&'static str> = if has_workflow_model_resources {
                            None
                        } else {
                            worker_tx.send(AppEvent::StatusUpdate("Detecting architecture...".to_string()))?;
                            let arch_name = gguf
                                .metadata()
                                .entries
                                .get("general.architecture")
                                .expect("Architecture not found in GGUF metadata")
                                .clone();

                            worker_tx.send(AppEvent::StatusUpdate(format!("Detected architecture: {:?}", arch_name)))?;

                            // Determine spec file path
                            if let metallic_context::gguf::GGUFValue::String(arch) = arch_name.clone() {
                                if arch.contains("qwen2") {
                                    Some("models/qwen25.json")
                                } else {
                                    return Err(anyhow::anyhow!("Unsupported architecture for Foundry: {:?}", arch_name));
                                }
                            } else {
                                return Err(anyhow::anyhow!(
                                    "Architecture not listed in GGUF metadata for Foundry: {:?}",
                                    arch_name
                                ));
                            }
                        };

                        worker_tx.send(AppEvent::StatusUpdate("Initializing Foundry...".to_string()))?;
                        let mut foundry = metallic_foundry::Foundry::new()?;

                        worker_tx.send(AppEvent::StatusUpdate("Building compiled model(s)...".to_string()))?;

                        let mut models_owned: rustc_hash::FxHashMap<String, Arc<CompiledModel>> = rustc_hash::FxHashMap::default();

                        if has_workflow_model_resources {
                            let resources = workflow_override
                                .as_ref()
                                .and_then(|w| w.resources.as_ref())
                                .expect("has_workflow_model_resources implies resources");
                            for m in &resources.models {
                                let model = metallic_foundry::model::ModelBuilder::new()
                                    .with_spec_file(std::path::PathBuf::from(&m.spec_path))?
                                    .with_gguf(&m.gguf_path)?
                                    .build(&mut foundry)?;
                                models_owned.insert(m.id.clone(), Arc::new(model));
                            }
                        } else {
                            let spec_path = spec_path.expect("spec_path required for single-model Foundry");
                            let model = metallic_foundry::model::ModelBuilder::new()
                                .with_spec_file(std::path::PathBuf::from(spec_path))?
                                .with_gguf(&gguf_path)?
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
                        let tokenization_start = std::time::Instant::now();
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

                        let mut cfg = metallic_foundry::generation::GenerationConfig {
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

                        // NOTE: repetition penalties are currently only correct for single-token decode.
                        // If the user enables any form of batching, we force-disable repetition penalties to
                        // avoid incorrect behavior (e.g., unstable repetition / quality regressions).
                        let workflow_batch = env_usize("METALLIC_WORKFLOW_BATCH_SIZE");
                        let decode_batch = env_usize("METALLIC_FOUNDRY_DECODE_BATCH_SIZE");
                        let max_batch = workflow_batch.into_iter().chain(decode_batch).max().unwrap_or(1);
                        if max_batch > 1
                            && (cfg.repeat_last_n > 0
                                || cfg.repeat_penalty != 1.0
                                || cfg.presence_penalty != 0.0
                                || cfg.frequency_penalty != 0.0)
                        {
                            tracing::warn!(
                                "Batch size > 1 detected (METALLIC_WORKFLOW_BATCH_SIZE={:?}, METALLIC_FOUNDRY_DECODE_BATCH_SIZE={:?}); disabling repetition/presence/frequency penalties for correctness.",
                                env_usize("METALLIC_WORKFLOW_BATCH_SIZE"),
                                env_usize("METALLIC_FOUNDRY_DECODE_BATCH_SIZE"),
                            );
                            cfg.repeat_penalty = 1.0;
                            cfg.repeat_last_n = 0;
                            cfg.presence_penalty = 0.0;
                            cfg.frequency_penalty = 0.0;
                        }

                        // `models_owned` is already FxHashMap<String, Arc<CompiledModel>>.
                        // We used to create a map of references, but now `generate_streaming_from_tokens_with_workflow`
                        // takes `&FxHashMap<String, Arc<CompiledModel>>`.
                        let models = models_owned; // Ownership transfer / move for clarity, though we could just use models_owned.

                        if !interactive {
                            worker_tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
                            if debug_tokenize {
                                let formatted = if disable_chat_template {
                                    prompts[0].clone()
                                } else {
                                    tokenizer.format_single_turn_chat_prompt(&prompts[0])?
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
                            }

                            let tokens0 = if disable_chat_template {
                                tokenizer.encode(&prompts[0])?
                            } else {
                                tokenizer.encode_single_turn_chat_prompt(&prompts[0])?
                            };

                            let tokenization_duration = tokenization_start.elapsed();
                            worker_tx.send(AppEvent::TokenizationComplete(tokenization_duration))?;

                            worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;

                            if workflow_wants_messages {
                                if prompts.len() > 1 {
                                    return Err(anyhow::anyhow!(
                                        "Workflow declares 'messages' input (prompt-driven workflow). Multi-turn CLI (--prompts ...) is not yet supported for prompt-driven workflows."
                                    ));
                                }
                                if prompts.first().map(|s| s.trim()).unwrap_or("").is_empty() {
                                    return Err(anyhow::anyhow!(
                                        "Workflow declares 'messages' input (prompt-driven workflow) but no prompt was provided. Pass a prompt argument, or use `--output-format tui` for interactive chat."
                                    ));
                                }

                                worker_tx.send(AppEvent::TokenCount(tokens0.len()))?;
                                let workflow = workflow_override.as_ref().expect("workflow_override present");
                                metallic_foundry::generation::generate_streaming_with_workflow_from_prompt(
                                    &mut foundry,
                                    &models,
                                    &tokenizer,
                                    &prompts[0],
                                    &cfg,
                                    &worker_tx,
                                    workflow.clone(),
                                )?;
                                return Ok(());
                            }

                            for (turn_idx, turn_prompt) in prompts.iter().enumerate() {
                                let mut current_tokens = if turn_idx == 0 {
                                    tokens0.clone()
                                } else if disable_chat_template {
                                    tokenizer.encode(turn_prompt)?
                                } else {
                                    tokenizer.encode_chat_continuation_prompt(turn_prompt)?
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
                                if let Some(workflow) = &workflow_override {
                                    metallic_foundry::generation::generate_streaming_from_tokens_with_workflow(
                                        &mut foundry,
                                        &models,
                                        &tokenizer,
                                        &current_tokens,
                                        &cfg,
                                        &worker_tx,
                                        workflow.clone(),
                                    )?;
                                } else {
                                    // Single-model default workflow path.
                                    let model_id = "llm";
                                    let model = models
                                        .get(model_id)
                                        .cloned()
                                        .or_else(|| models.values().next().cloned())
                                        .expect("at least one model");
                                    metallic_foundry::generation::generate_streaming_from_tokens(
                                        &mut foundry,
                                        model,
                                        &tokenizer,
                                        &current_tokens,
                                        &cfg,
                                        &worker_tx,
                                    )?;
                                }
                            }
                        } else if workflow_wants_messages {
                            // TUI multi-turn for message-driven workflows:
                            // - Feed only user-turn deltas into the workflow to avoid re-prefilling assistant text.
                            // - Preserve KV cache in Foundry sessions; prefill consumes only the delta tokens.
                            use metallic_foundry::workflow::Value as WfValue;

                            fn sys_prompt() -> String {
                                std::env::var("METALLIC_SYSTEM_PROMPT")
                                    .ok()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| "You are a helpful assistant.".to_string())
                            }

                            fn msg(role: &str, content: &str) -> WfValue {
                                let mut map = rustc_hash::FxHashMap::default();
                                map.insert("role".to_string(), WfValue::Text(role.into()));
                                map.insert("content".to_string(), WfValue::Text(content.to_string().into()));
                                WfValue::Map(map)
                            }

                            let workflow = workflow_override.as_ref().expect("workflow_override present");
                            let mut runner = metallic_foundry::workflow::WorkflowRunner::new(&mut foundry, models);

                            let system = sys_prompt();
                            let mut is_first_turn = true;

                            // If the user provided multiple prompts on the CLI for a TUI session, treat them
                            // as queued turns.
                            let mut queued_cli_turns = prompts.iter().skip(1);

                            if debug_tokenize && !first_prompt.is_empty() {
                                let formatted = if disable_chat_template {
                                    prompts[0].clone()
                                } else {
                                    tokenizer.format_single_turn_chat_prompt(&prompts[0])?
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
                                    vec![msg("system", &system), msg("user", &user_prompt)]
                                } else {
                                    vec![msg("user", &user_prompt)]
                                };
                                is_first_turn = false;

                                worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;

                                let mut inputs: rustc_hash::FxHashMap<String, WfValue> = rustc_hash::FxHashMap::default();
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
                                let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
                                inputs.insert("eos_token".to_string(), WfValue::U32(eos));

                                let mut decode_scratch = Vec::new();
                                let mut decoded_chunk = String::new();
                                let mut on_token = |token_id: u32,
                                                    prefill: std::time::Duration,
                                                    setup: std::time::Duration,
                                                    iter: Option<std::time::Duration>|
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

                                let gen_start = std::time::Instant::now();
                                let _outputs = runner.run_streaming(workflow, inputs, &mut on_token)?;
                                let _ = worker_tx.send(AppEvent::GenerationComplete {
                                    total_generation_time: gen_start.elapsed(),
                                });
                            }
                        } else {
                            let mut is_first_turn = true;
                            let mut current_tokens: Vec<u32> = if !first_prompt.is_empty() {
                                is_first_turn = false;
                                let toks = if disable_chat_template {
                                    tokenizer.encode(&prompts[0])?
                                } else {
                                    tokenizer.encode_single_turn_chat_prompt(&prompts[0])?
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
                                                tokenizer.encode_single_turn_chat_prompt(turn_prompt)?
                                            }
                                        } else {
                                            remove_bos = true;
                                            tokenizer.encode_chat_continuation_prompt(turn_prompt)?
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
                                                    tokenizer.encode_single_turn_chat_prompt(&input)?
                                                }
                                            } else {
                                                remove_bos = true;
                                                tokenizer.encode_chat_continuation_prompt(&input)?
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

                                if let Some(workflow) = &workflow_override {
                                    metallic_foundry::generation::generate_streaming_from_tokens_with_workflow(
                                        &mut foundry,
                                        &models,
                                        &tokenizer,
                                        &current_tokens,
                                        &cfg,
                                        &worker_tx,
                                        workflow.clone(),
                                    )?;
                                } else {
                                    let model_id = "llm";
                                    let model = models
                                        .get(model_id)
                                        .cloned()
                                        .or_else(|| models.values().next().cloned())
                                        .expect("at least one model");
                                    metallic_foundry::generation::generate_streaming_from_tokens(
                                        &mut foundry,
                                        model,
                                        &tokenizer,
                                        &current_tokens,
                                        &cfg,
                                        &worker_tx,
                                    )?;
                                }

                                // If the user provided multiple prompts on the CLI for a TUI session, consume them
                                // as queued user turns before switching to interactive input.
                                if let Some(turn_prompt) = queued_cli_turns.next() {
                                    // Echo queued CLI prompts into the transcript so interactive users don't see the chat "lag"
                                    // behind (the model will answer this queued prompt next).
                                    worker_tx.send(AppEvent::UserPrompt(turn_prompt.to_string()))?;
                                    worker_tx.send(AppEvent::StatusUpdate("Processing queued prompt...".to_string()))?;
                                    current_tokens = tokenizer.encode_chat_continuation_prompt(turn_prompt)?;

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
                                        current_tokens = tokenizer.encode_chat_continuation_prompt(&input)?;

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
    };

    // Based on output format, run the appropriate mode
    match cli_config.output_format {
        cli::config::OutputFormat::Tui => run_tui_mode(&async_recorder.receiver, &rx, &cmd_tx, tui_start_processing, generation_handle)?,
        cli::config::OutputFormat::Text | cli::config::OutputFormat::None => {
            run_text_mode(&cli_config, &async_recorder.receiver, &rx, generation_handle)?
        }
        cli::config::OutputFormat::Json => run_json_mode(&async_recorder.receiver, &rx, generation_handle)?,
    }

    Ok(())
}

fn report_model_weight_breakdown(qwen: &metallic_context::models::Qwen25<F16Element>) {
    // Report model weights breakdown
    let mut breakdown = FxHashMap::default();
    let mut total_weights_size = 0u64;
    let bytes_per_element = F16Element::DTYPE.size_bytes();

    // Token Embeddings
    let embed_size = (qwen.embed_weight.len() * bytes_per_element) as u64;
    breakdown.insert("Token Embeddings".to_string(), embed_size);
    total_weights_size += embed_size;

    // Output Projection
    let output_size = (qwen.output_weight.as_ref().map(|w| w.len()).unwrap_or(0) * bytes_per_element
        + qwen.output_weight_canon.as_ref().map(|w| w.len()).unwrap_or(0) * bytes_per_element) as u64;
    breakdown.insert("Output Projection".to_string(), output_size);
    total_weights_size += output_size;

    // Final Layer Norm
    let norm_size = (qwen.final_norm_gamma.len() * bytes_per_element) as u64;
    breakdown.insert("Final Layer Norm".to_string(), norm_size);
    total_weights_size += norm_size;

    // RoPE Cache
    let rope_cache_size = (qwen.rope_cos_cache.len() * bytes_per_element + qwen.rope_sin_cache.len() * bytes_per_element) as u64;
    breakdown.insert("RoPE Cache".to_string(), rope_cache_size);
    total_weights_size += rope_cache_size;

    // Transformer Blocks
    let mut total_transformer_blocks_size = 0u64;
    for (i, block) in qwen.blocks.iter().enumerate() {
        let block_base_key = format!("Transformer Blocks.Weight Block {}", i + 1);

        // Attention Projections
        let legacy_qkv_size = block.attn_qkv_weight.as_ref().map(|w| w.len()).unwrap_or(0);
        let canon_q_size = block.attn_q_weight_canon.as_ref().map(|w| w.len()).unwrap_or(0);
        let canon_k_size = block.attn_k_weight_canon.as_ref().map(|w| w.len()).unwrap_or(0);
        let canon_v_size = block.attn_v_weight_canon.as_ref().map(|w| w.len()).unwrap_or(0);
        let fused_qkv_weight_size = ((legacy_qkv_size + canon_q_size + canon_k_size + canon_v_size) * bytes_per_element) as u64;
        breakdown.insert(
            format!("{}.Attention Projections.Fused QKV weight", block_base_key),
            fused_qkv_weight_size,
        );
        let output_weight_size = ((block.attn_out_weight.as_ref().map(|w| w.len()).unwrap_or(0)
            + block.attn_out_weight_canon.as_ref().map(|w| w.len()).unwrap_or(0))
            * bytes_per_element) as u64;
        breakdown.insert(
            format!("{}.Attention Projections.Output weight", block_base_key),
            output_weight_size,
        );
        let total_attn_proj_size = fused_qkv_weight_size + output_weight_size;
        breakdown.insert(format!("{}.Attention Projections", block_base_key), total_attn_proj_size);

        // Attention Biases
        let fused_qkv_bias_size = (block.attn_qkv_bias.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Attention Biases.Fused QKV bias", block_base_key), fused_qkv_bias_size);
        breakdown.insert(format!("{}.Attention Biases", block_base_key), fused_qkv_bias_size);

        // Feedforward Projections
        let gate_weight_size = ((block.ffn_gate.as_ref().map(|w| w.len()).unwrap_or(0)
            + block.ffn_gate_canon.as_ref().map(|w| w.len()).unwrap_or(0))
            * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Projections.Gate weight", block_base_key), gate_weight_size);
        let up_weight_size = ((block.ffn_up.as_ref().map(|w| w.len()).unwrap_or(0)
            + block.ffn_up_canon.as_ref().map(|w| w.len()).unwrap_or(0))
            * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Projections.Up weight", block_base_key), up_weight_size);
        let down_weight_size = ((block.ffn_down.as_ref().map(|w| w.len()).unwrap_or(0)
            + block.ffn_down_canon.as_ref().map(|w| w.len()).unwrap_or(0))
            * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Projections.Down weight", block_base_key), down_weight_size);
        let total_ffn_proj_size = gate_weight_size + up_weight_size + down_weight_size;
        breakdown.insert(format!("{}.Feedforward Projections", block_base_key), total_ffn_proj_size);

        // Feedforward Biases
        let gate_bias_size = (block.ffn_gate_bias.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Biases.Gate bias", block_base_key), gate_bias_size);
        let up_bias_size = (block.ffn_up_bias.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Biases.Up bias", block_base_key), up_bias_size);
        let down_bias_size = (block.ffn_down_bias.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Feedforward Biases.Down bias", block_base_key), down_bias_size);
        let total_ffn_bias_size = gate_bias_size + up_bias_size + down_bias_size;
        breakdown.insert(format!("{}.Feedforward Biases", block_base_key), total_ffn_bias_size);

        // Norm Parameters
        let attn_norm_size = (block.attn_norm_gamma.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Norm Parameters.Attention norm", block_base_key), attn_norm_size);
        let ffn_norm_size = (block.ffn_norm_gamma.len() * bytes_per_element) as u64;
        breakdown.insert(format!("{}.Norm Parameters.FFN norm", block_base_key), ffn_norm_size);
        let total_norm_param_size = attn_norm_size + ffn_norm_size;
        breakdown.insert(format!("{}.Norm Parameters", block_base_key), total_norm_param_size);

        let total_block_size =
            total_attn_proj_size + fused_qkv_bias_size + total_ffn_proj_size + total_ffn_bias_size + total_norm_param_size;
        breakdown.insert(block_base_key, total_block_size);
        total_transformer_blocks_size += total_block_size;
    }
    breakdown.insert("Transformer Blocks".to_string(), total_transformer_blocks_size);
    total_weights_size += total_transformer_blocks_size;

    record_metric_async!(MetricEvent::ModelWeights {
        total_bytes: total_weights_size,
        breakdown,
    });
}

fn run_tui_mode(
    receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    cmd_tx: &std::sync::mpsc::Sender<AppEvent>,
    start_processing: bool,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    app.is_processing = start_processing;

    // Get the initial profiling state and set it in the app
    let initial_profiling_state = profiling_state::get_profiling_state();
    app.set_profiling_active(initial_profiling_state);

    while !app.should_quit {
        // Handle crossterm events
        if crossterm::event::poll(std::time::Duration::from_millis(50))? {
            match crossterm::event::read()? {
                CrosstermEvent::Key(key) => {
                    if key.code == crossterm::event::KeyCode::Char('p') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                        // Toggle profiling state
                        profiling_state::toggle_profiling_state();
                        metallic_foundry::instrument::toggle_profiling_state();
                        let new_state = profiling_state::get_profiling_state();
                        app.set_profiling_active(new_state);
                    } else {
                        // Handle Input Mode
                        let handled = if app.focus == FocusArea::Input {
                            match key.code {
                                crossterm::event::KeyCode::Char(c) if !key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                    app.input_buffer.push(c);
                                    true
                                }
                                crossterm::event::KeyCode::Backspace => {
                                    app.input_buffer.pop();
                                    true
                                }
                                crossterm::event::KeyCode::Enter => {
                                    if !app.input_buffer.trim().is_empty() {
                                        if app.is_processing {
                                            app.push_alert(Alert::warning(
                                                "Still generating; wait for completion before sending the next message.",
                                            ));
                                            app.input_buffer.clear();
                                        } else {
                                            let input = app.input_buffer.clone();
                                            app.input_buffer.clear();

                                            // Echo to UI
                                            if !app.generated_text.is_empty() && !app.generated_text.ends_with("\n\n") {
                                                app.generated_text.push_str("\n\n");
                                            }
                                            app.generated_text.push_str(&format!("> {}\n\n", input));

                                            app.is_processing = true;
                                            app.scroll_text_to_end();

                                            let _ = cmd_tx.send(AppEvent::Input(input));
                                        }
                                    }
                                    true
                                }
                                crossterm::event::KeyCode::Esc => {
                                    app.focus = FocusArea::GeneratedText;
                                    true
                                }
                                _ => false,
                            }
                        } else {
                            false
                        };

                        if !handled {
                            match key.code {
                                crossterm::event::KeyCode::Char('q') => app.quit(),
                                crossterm::event::KeyCode::Enter
                                | crossterm::event::KeyCode::Char(' ')
                                | crossterm::event::KeyCode::Esc => {
                                    if app.has_active_alert() {
                                        app.dismiss_active_alert();
                                    }
                                }
                                crossterm::event::KeyCode::Char('m') => {
                                    app.metrics_view = tui::app::MetricsView::Memory;
                                    app.reset_metrics_scroll();
                                }
                                crossterm::event::KeyCode::Char('l') => {
                                    if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                        app.toggle_log_visibility();
                                    } else {
                                        app.metrics_view = tui::app::MetricsView::Latency;
                                        app.reset_metrics_scroll();
                                    }
                                }
                                crossterm::event::KeyCode::Char('s') => {
                                    app.metrics_view = tui::app::MetricsView::Stats;
                                    app.reset_metrics_scroll();
                                }
                                crossterm::event::KeyCode::Char('c') => {
                                    if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                        let content_to_copy = match app.focus {
                                            FocusArea::GeneratedText => app.generated_text.clone(),
                                            FocusArea::Metrics => {
                                                // ... (reused existing logic for metrics string gen would be ideal, but repeating is safer for update)
                                                // I'll copy the block content from the file to avoid error
                                                let metrics_help = match app.metrics_view {
                                                    tui::app::MetricsView::Memory => "[m] Memory [l] Latency [s] Stats [c] Collapse",
                                                    tui::app::MetricsView::Latency => "[m] Memory [l] Latency [s] Stats [c] Collapse",
                                                    tui::app::MetricsView::Stats => "[m] Memory [l] Latency [s] Stats [c] Collapse",
                                                };
                                                let metrics_content = match app.metrics_view {
                                                    tui::app::MetricsView::Memory => ui::render_memory_metrics(
                                                        &app.memory_rows,
                                                        app.memory_collapse_depth.get_current_depth(),
                                                    ),
                                                    tui::app::MetricsView::Latency => ui::render_hierarchical_latency_metrics(
                                                        &app.latency_tree,
                                                        app.latency_collapse_depth.get_current_depth(),
                                                    ),
                                                    tui::app::MetricsView::Stats => ui::render_stats_metrics_from_app(&app),
                                                };
                                                format!("{}\n\n{}", metrics_help, metrics_content)
                                            }
                                            FocusArea::LogBox => app.log_messages.join("\n"),
                                            FocusArea::Input => app.input_buffer.clone(),
                                        };
                                        copy_text_to_clipboard(&content_to_copy);
                                    } else {
                                        app.toggle_collapse();
                                    }
                                }
                                crossterm::event::KeyCode::Tab => {
                                    app.focus_next();
                                    app.reset_metrics_scroll();
                                }
                                crossterm::event::KeyCode::Char('d') => {
                                    if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                        app.text_display_mode = match app.text_display_mode {
                                            tui::app::TextDisplayMode::Plain => tui::app::TextDisplayMode::Markdown,
                                            tui::app::TextDisplayMode::Markdown => tui::app::TextDisplayMode::Plain,
                                        };
                                    }
                                }
                                crossterm::event::KeyCode::Up => app.scroll_active(-1),
                                crossterm::event::KeyCode::Down => app.scroll_active(1),
                                crossterm::event::KeyCode::PageUp => app.scroll_active(-10),
                                crossterm::event::KeyCode::PageDown => app.scroll_active(10),
                                crossterm::event::KeyCode::Home => app.scroll_active_to_start(),
                                crossterm::event::KeyCode::End => app.scroll_active_to_end(),
                                _ => {}
                            }
                        }
                    }
                }
                CrosstermEvent::Mouse(mouse_event) => {
                    handle_mouse_event(mouse_event, &mut app);
                }
                _ => {}
            }
        }

        // Process metric events from the instrumentation system and convert them to AppEvents
        while let Ok(enriched_event) = receiver.try_recv() {
            tracing::debug!("Main thread received enriched event: {:?}", enriched_event);

            // Log the metric event to the log box
            if let Ok(serialised) = serde_json::to_string(&enriched_event.event) {
                let log_message = format!("METRIC: {}", serialised);
                handle_app_event(&mut app, AppEvent::LogMessage(log_message));
            }

            let latency_rows = tui::metrics::metric_event_to_latency_rows(&enriched_event.event);
            tracing::debug!("Converted to {} latency rows", latency_rows.len());
            if !latency_rows.is_empty() {
                // Send the rows as an AppEvent so they get processed like before
                handle_app_event(&mut app, AppEvent::LatencyUpdate(latency_rows));
            }

            // Process memory events
            let memory_rows = tui::metrics::metric_event_to_memory_rows(&enriched_event.event);
            tracing::debug!("Converted to {} memory rows", memory_rows.len());
            if !memory_rows.is_empty() {
                handle_app_event(&mut app, AppEvent::MemoryUpdate(memory_rows));
            }

            // Process stats events
            let stats_rows = tui::metrics::metric_event_to_stats_rows(&enriched_event.event);
            tracing::debug!("Converted to {} stats rows", stats_rows.len());
            if !stats_rows.is_empty() {
                handle_app_event(&mut app, AppEvent::StatsUpdate(stats_rows));
            }
        }

        // Process app events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::LogMessage(message) => {
                    // Only add log message to UI if log view is visible (for performance)
                    if app.log_visible {
                        app.add_log_message(&message);
                    }
                }
                _ => handle_app_event(&mut app, event),
            }
        }

        terminal.draw(|frame| tui::ui::render(&mut app, frame))?;
    }

    restore_terminal()?;
    generation_handle.join().unwrap()?;
    Ok(())
}

fn run_text_mode(
    cli_config: &cli::CliConfig,
    metrics_receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    let total_turns = cli_config.get_prompts().len();
    let mut turns_completed: usize = 0;

    let mut generated_tokens: u64 = 0;
    let mut prompt_processing: Option<Duration> = None;
    let mut setup_duration: Option<Duration> = None;
    let mut total_generation_time: Option<Duration> = None;
    let mut tokenization_time: Option<Duration> = None;
    let mut model_load_time: Option<Duration> = None;
    let mut prompt_token_count: usize = 0;

    // Process all events until generation is done
    while !generation_handle.is_finished() {
        // Drain metrics channel to prevent memory leak and unnecessary buffering overhead
        while let Ok(_metric) = metrics_receiver.try_recv() {}

        // Process any pending app events
        // We use recv_timeout to avoid busy waiting but still check for generation_handle finishing frequently
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(event) => process_text_mode_event(
                event,
                cli_config,
                total_turns,
                &mut turns_completed,
                &mut generated_tokens,
                &mut setup_duration,
                &mut prompt_processing,
                &mut total_generation_time,
                &mut model_load_time,
                &mut tokenization_time,
                &mut prompt_token_count,
            ),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Just continue to check loop condition
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Channel closed, so generation must be done (or crashed)
                break;
            }
        }
    }

    // Process any remaining events
    while let Ok(event) = rx.try_recv() {
        process_text_mode_event(
            event,
            cli_config,
            total_turns,
            &mut turns_completed,
            &mut generated_tokens,
            &mut setup_duration,
            &mut prompt_processing,
            &mut total_generation_time,
            &mut model_load_time,
            &mut tokenization_time,
            &mut prompt_token_count,
        );
    }

    generation_handle.join().unwrap()?;

    if std::env::var("METALLIC_PERF_OUTPUT").is_ok()
        && let Some(total) = total_generation_time
    {
        // Flush stdout so we don't interleave with the perf report if it goes to stderr?
        // Actually perf report goes to stderr usually to separate from output.
        // Let's ensure a newline first just in case.
        if generated_tokens > 0 {
            println!();
        }

        eprintln!("\n[metallic] Performance Breakdown:");

        if let Some(load_time) = model_load_time {
            eprintln!("[metallic]   Model Load:        {:.3}s", load_time.as_secs_f64());
        }

        if let Some(tok_time) = tokenization_time {
            let tok_s = tok_time.as_secs_f64().max(1e-9);
            let tok_speed = if prompt_token_count > 0 {
                prompt_token_count as f64 / tok_s
            } else {
                0.0
            };
            eprintln!("[metallic]   Tokenization:      {:.3}s ({:.2} tokens/s)", tok_s, tok_speed);
        }

        if let Some(prompt) = prompt_processing {
            let prompt_s = prompt.as_secs_f64().max(1e-9);
            let pp_tok_s = if prompt_token_count > 0 {
                prompt_token_count as f64 / prompt_s
            } else {
                0.0
            };
            eprintln!("[metallic]   Prompt Processing: {:.3}s ({:.2} tok/s)", prompt_s, pp_tok_s);

            if let Some(setup) = setup_duration {
                eprintln!("[metallic]   Setup:             {:.3}s", setup.as_secs_f64());
            }

            let mut decode_time = total.saturating_sub(prompt);
            if let Some(setup) = setup_duration {
                decode_time = decode_time.saturating_sub(setup);
            }
            let decode_s = decode_time.as_secs_f64().max(1e-9);
            let tps_decode = (generated_tokens as f64) / decode_s;
            eprintln!("[metallic]   Decode:            {:.3}s ({:.2} tokens/s)", decode_s, tps_decode);
        }

        let total_s = total.as_secs_f64().max(1e-9);
        let tps_total = (generated_tokens as f64) / total_s;
        eprintln!("[metallic]   Total:             {:.3}s ({:.2} tokens/s)", total_s, tps_total);

        // End-to-end excludes Cargo build/launch but includes in-process model load + tokenization + generation.
        if let Some(load) = model_load_time {
            let tok = tokenization_time.unwrap_or(Duration::ZERO);
            let e2e = load.saturating_add(tok).saturating_add(total);
            let e2e_s = e2e.as_secs_f64().max(1e-9);
            let e2e_tps = (generated_tokens as f64) / e2e_s;
            eprintln!("[metallic]   End-to-End:        {:.3}s ({:.2} tokens/s)", e2e_s, e2e_tps);
        }
        eprintln!("[metallic]   Tokens Generated:  {}", generated_tokens);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_text_mode_event(
    event: AppEvent,
    cli_config: &cli::CliConfig,
    total_turns: usize,
    turns_completed: &mut usize,
    generated_tokens: &mut u64,
    setup_duration: &mut Option<Duration>,
    prompt_processing: &mut Option<Duration>,
    total_generation_time: &mut Option<Duration>,
    model_load_time: &mut Option<Duration>,
    tokenization_time: &mut Option<Duration>,
    prompt_token_count: &mut usize,
) {
    match event {
        AppEvent::Token {
            text,
            setup_duration: setup,
            prompt_processing: prompt,
            ..
        } => {
            if !matches!(cli_config.output_format, cli::config::OutputFormat::None) {
                print!("{}", text);
                // Only flush on newlines to reduce terminal I/O and GPU contention
                if text.contains('\n') || (*generated_tokens).is_multiple_of(16) {
                    std::io::stdout().flush().unwrap();
                }
            }
            *generated_tokens = generated_tokens.saturating_add(1);
            if let Some(setup) = setup
                && !setup.is_zero()
            {
                setup_duration.get_or_insert(setup);
            }
            prompt_processing.get_or_insert(prompt);
        }
        AppEvent::GenerationComplete {
            total_generation_time: total,
        } => {
            *total_generation_time = Some(total);
            if !matches!(cli_config.output_format, cli::config::OutputFormat::None) {
                // Ensure output ends with a newline so shells don't render a trailing `%`.
                println!();
                let _ = std::io::stdout().flush();
            }
            if total_turns > 1 {
                *turns_completed = turns_completed.saturating_add(1);
                if *turns_completed < total_turns && !matches!(cli_config.output_format, cli::config::OutputFormat::None) {
                    print!("\n\n");
                    let _ = std::io::stdout().flush();
                }
            }
        }
        AppEvent::ModelLoadComplete(duration) => {
            *model_load_time = Some(duration);
            tracing::debug!("Model load time: {:?}", duration);
        }
        AppEvent::TokenizationComplete(duration) => {
            *tokenization_time = Some(duration);
            tracing::debug!("Tokenization time: {:?}", duration);
        }
        AppEvent::StatusUpdate(status) => {
            // Only log status updates if we're in verbose mode
            // For now, using debug level which won't show by default
            tracing::debug!("Status update: {}", status);
        }
        AppEvent::Alert(alert) => match alert.level {
            metallic_cli_helpers::app_event::AlertLevel::Info => {
                tracing::info!("{}", alert.message);
            }
            metallic_cli_helpers::app_event::AlertLevel::Warning => {
                tracing::warn!("{}", alert.message);
            }
            metallic_cli_helpers::app_event::AlertLevel::Error => {
                tracing::error!("{}", alert.message);
            }
        },
        AppEvent::TokenCount(count) => {
            *prompt_token_count = count;
            tracing::debug!("Prompt token count: {}", count);
        }
        AppEvent::Input(_) => {}
        _ => {} // Ignore other events in text mode
    }
}

fn run_json_mode(
    _receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    use tracing::{debug, error, info, warn};
    let mut full_text = String::new();
    let mut logs = Vec::new();

    // Process all events until generation is done
    while !generation_handle.is_finished() || rx.recv_timeout(Duration::from_millis(100)).is_ok() {
        // Process any pending app events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token { text, .. } => {
                    full_text.push_str(&text);
                }
                AppEvent::StatusUpdate(status) => {
                    let log_entry = serde_json::json!({
                        "type": "status",
                        "status": status,
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                    debug!("Status update: {}", status);
                }
                AppEvent::Alert(alert) => {
                    match alert.level {
                        metallic_cli_helpers::app_event::AlertLevel::Info => {
                            info!("{}", alert.message);
                        }
                        metallic_cli_helpers::app_event::AlertLevel::Warning => {
                            warn!("{}", alert.message);
                        }
                        metallic_cli_helpers::app_event::AlertLevel::Error => {
                            error!("{}", alert.message);
                        }
                    }
                    let log_entry = serde_json::json!({
                        "type": "alert",
                        "level": alert.level.as_str(),
                        "message": alert.message,
                        "timestamp": alert.timestamp.to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::TokenCount(count) => {
                    let log_entry = serde_json::json!({
                        "type": "token_count",
                        "count": count
                    });
                    logs.push(log_entry);
                    debug!("Prompt token count: {}", count);
                }
                AppEvent::MemoryUpdate(memory_rows) => {
                    let log_entry = serde_json::json!({
                        "type": "memory_update",
                        "rows": memory_rows
                    });
                    logs.push(log_entry);
                }
                AppEvent::LatencyUpdate(latency_rows) => {
                    let log_entry = serde_json::json!({
                        "type": "latency_update",
                        "rows": latency_rows
                    });
                    logs.push(log_entry);
                }
                AppEvent::StatsUpdate(stats_rows) => {
                    let log_entry = serde_json::json!({
                        "type": "stats_update",
                        "rows": stats_rows
                    });
                    logs.push(log_entry);
                }
                AppEvent::GenerationComplete { total_generation_time: _ } => {
                    // For JSON output mode, we just log that generation completed
                    let log_entry = serde_json::json!({
                        "type": "generation_complete",
                        "message": "Generation completed successfully",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::ModelLoadComplete(duration) => {
                    let log_entry = serde_json::json!({
                        "type": "model_load_complete",
                        "duration_ms": duration.as_millis(),
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::TokenizationComplete(duration) => {
                    let log_entry = serde_json::json!({
                        "type": "tokenization_complete",
                        "duration_ms": duration.as_millis(),
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::LogMessage(message) => {
                    let log_entry = serde_json::json!({
                        "type": "log_message",
                        "message": message,
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::UserPrompt(prompt) => {
                    let log_entry = serde_json::json!({
                        "type": "user_prompt",
                        "prompt": prompt,
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    logs.push(log_entry);
                }
                AppEvent::Input(_) => {}
            }
        }
    }

    // Output the final result with all collected logs
    let json_output = serde_json::json!({
        "type": "completion",
        "text": full_text,
        "logs": logs,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    println!("{}", serde_json::to_string(&json_output)?);

    generation_handle.join().unwrap()?;
    Ok(())
}

fn handle_app_event(app: &mut App, event: AppEvent) {
    match event {
        AppEvent::Token {
            text,
            setup_duration: _,
            prompt_processing,
            iteration,
        } => {
            let was_following = app.text_follow_bottom;
            app.generated_text.push_str(&text);
            if was_following {
                app.request_follow_text = true;
            }
            app.update_generation_metrics(iteration);
            app.prompt_processing_time = prompt_processing;
        }
        AppEvent::TokenCount(count) => {
            app.reset_generation_metrics();
            app.reset_prompt_processing_metrics();
            app.prompt_token_count = count;
            app.is_processing = true;
        }
        AppEvent::TokenizationComplete(_) => {
            // No specific TUI update for tokenization complete yet, but we need to handle the event
        }
        AppEvent::StatusUpdate(status) => {
            app.status = status;
        }
        AppEvent::MemoryUpdate(memory_rows) => {
            // Replace existing memory rows of the same type instead of accumulating
            // This prevents duplicate entries for the same memory metric
            let mut merged_rows = app.memory_rows.clone();

            for new_row in memory_rows {
                // Check if a row with the same label already exists
                if let Some(existing_index) = merged_rows.iter().position(|row| row.label == new_row.label) {
                    // Replace the existing row with the new one
                    merged_rows[existing_index] = new_row;
                } else {
                    // Add as a new row if it doesn't exist
                    merged_rows.push(new_row);
                }
            }

            app.memory_rows = merged_rows;
            // Recalculate max depth for memory metrics since the rows may have changed
            app.memory_collapse_depth.calculate_max_depth(&app.memory_rows);
        }
        AppEvent::LatencyUpdate(new_rows) => {
            // Process the incoming metrics and add them to our hierarchical structure
            for row in new_rows {
                app.add_latency_metric(row);
            }
        }
        AppEvent::StatsUpdate(stats_rows) => {
            // Determine the type of stats and update the appropriate field
            if !stats_rows.is_empty() {
                // Check if these are resource cache stats or tensor preparation stats
                if stats_rows[0].label.contains("Cache") && !stats_rows[0].label.contains("Tensor Preparation") {
                    // This is a resource cache stat - we need to find the actual cache type
                    // The first entry should be "{CacheType} Cache" at level 0 now (after our mapping change)
                    let cache_type = stats_rows[0].label.trim_start().to_string();
                    app.resource_cache_stats.insert(cache_type, stats_rows);
                } else {
                    // This is tensor preparation stats or other stats
                    app.tensor_preparation_stats = stats_rows;
                }
            }
        }
        AppEvent::Alert(alert) => {
            app.push_alert(alert);
        }
        AppEvent::LogMessage(message) => {
            app.add_log_message(&message);
        }
        AppEvent::GenerationComplete { total_generation_time } => {
            app.generation_time = total_generation_time;
            app.is_processing = false;
        }
        AppEvent::ModelLoadComplete(_) => {}
        AppEvent::Input(_) => {}
        AppEvent::UserPrompt(prompt) => {
            if !app.generated_text.is_empty() && !app.generated_text.ends_with("\n\n") {
                app.generated_text.push_str("\n\n");
            }
            app.generated_text.push_str(&format!("> {}\n\n", prompt));
            if app.text_follow_bottom {
                app.request_follow_text = true;
            }
        }
    }
}

fn emit_startup_memory_update(tx: &mpsc::Sender<AppEvent>) -> Result<(), mpsc::SendError<AppEvent>> {
    // Send empty memory rows as placeholder in the new system
    tx.send(AppEvent::MemoryUpdate(vec![]))
}

fn setup_terminal() -> Result<Terminal<impl Backend>> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(
        stdout(),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal() -> Result<()> {
    crossterm::execute!(
        stdout(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

fn handle_mouse_event(event: MouseEvent, app: &mut App) {
    let position = (event.column, event.row).into();

    match event.kind {
        event::MouseEventKind::Down(MouseButton::Left) => {
            // Check if the click is in one of the main focus areas
            if app.text_area.contains(position) {
                app.focus = FocusArea::GeneratedText;

                // Start text selection, converting screen coordinates to text content coordinates
                // Account for text wrapping and borders
                let relative_x = event.column.saturating_sub(app.text_area.x);
                let relative_y = event.row.saturating_sub(app.text_area.y);

                // Subtract 1 from relative_y to account for the border/title at the top of the text area
                let adjusted_y = relative_y.saturating_sub(1);

                // Convert visual coordinates to content coordinates considering text wrapping
                let wrap_width = if app.text_area.width > 2 {
                    app.text_area.width.saturating_sub(2) // width accounting for left/right borders
                } else {
                    1 // minimum width of 1 to avoid issues
                };
                let (content_row, content_col) = app.get_content_position_from_visual(
                    adjusted_y,      // visual row (relative to text area after removing border)
                    relative_x,      // visual column
                    app.text_scroll, // scroll offset
                    &app.generated_text,
                    wrap_width,
                );

                let relative_pos = Position::new(content_col as u16, content_row as u16);
                app.start_text_selection(relative_pos);
            } else if app.metrics_area.contains(position) {
                app.focus = FocusArea::Metrics;
            } else if app.log_visible && app.log_area.contains(position) {
                app.focus = FocusArea::LogBox;
            } else if app.input_area.contains(position) {
                app.focus = FocusArea::Input;
            }
        }
        event::MouseEventKind::Drag(MouseButton::Left) => {
            // Update text selection - allow dragging even if slightly outside the text area if we started inside
            if app.focus == FocusArea::GeneratedText && app.is_selecting {
                let relative_x = event.column.saturating_sub(app.text_area.x);
                let relative_y = event.row.saturating_sub(app.text_area.y);

                // Calculate if we're beyond the content area
                let lines: Vec<&str> = app.generated_text.lines().collect();
                if lines.is_empty() {
                    // If no content, just return early
                    return;
                }

                // Check if we're dragging beyond the bottom of content
                let wrap_width = if app.text_area.width > 2 {
                    app.text_area.width.saturating_sub(2) // width accounting for left/right borders
                } else {
                    1 // minimum width of 1 to avoid issues
                };

                // Calculate total visual lines for all content
                let mut total_visual_lines = 0u16;
                for line in &lines {
                    let visual_lines = app.count_visual_lines_for_content_line(line, wrap_width);
                    total_visual_lines += visual_lines as u16;
                }

                // If we're dragging beyond the content vertically, clamp to the end
                let adjusted_y = relative_y.saturating_sub(1);
                let absolute_visual_y = adjusted_y.saturating_add(app.text_scroll);

                if (absolute_visual_y as usize) >= total_visual_lines as usize {
                    // We're dragging beyond the content, set to the end position
                    if let Some(last_line) = lines.last() {
                        let last_line_chars: Vec<char> = last_line.chars().collect();
                        let last_line_idx = lines.len() - 1;
                        let relative_pos = Position::new(last_line_chars.len() as u16, last_line_idx as u16);
                        app.update_text_selection(relative_pos);
                    }
                } else {
                    // Normal case: convert visual coordinates to content coordinates considering text wrapping
                    let (content_row, content_col) = app.get_content_position_from_visual(
                        adjusted_y,      // visual row
                        relative_x,      // visual column
                        app.text_scroll, // scroll offset
                        &app.generated_text,
                        wrap_width,
                    );

                    let relative_pos = Position::new(content_col as u16, content_row as u16);
                    app.update_text_selection(relative_pos);
                }
            }
        }
        event::MouseEventKind::Up(MouseButton::Left) => {
            // End text selection and copy to clipboard if there's a selection
            if app.focus == FocusArea::GeneratedText && app.is_selecting {
                app.end_text_selection();

                // Copy selected text to clipboard if there's a selection
                let selected_text = app.get_selected_text(&app.generated_text);
                if !selected_text.trim().is_empty() {
                    copy_text_to_clipboard(&selected_text);
                }
            }
        }
        event::MouseEventKind::ScrollUp => {
            app.scroll_active(-1);
        }
        event::MouseEventKind::ScrollDown => {
            app.scroll_active(1);
        }
        _ => {}
    }
}

fn copy_text_to_clipboard(text: &str) {
    // Attempt to copy text to clipboard using arboard - copy the text to avoid lifetime issues
    let text = text.to_string();
    std::thread::spawn(move || {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    });
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "unknown panic".to_string()
    }
}
