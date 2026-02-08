//! Inference engine integration for the GUI.
//!
//! This module handles model loading, inference execution, and KV cache management.

use std::{
    path::{Path, PathBuf}, sync::{
        Arc, Mutex, atomic::{AtomicBool, Ordering}
    }
};

use anyhow::Result;
use metallic_foundry::{
    Foundry, model::{CompiledModel, ModelBuilder}, model_routing::resolve_model_routing_from_loaded_model, workflow::{Value, WorkflowRunner, WorkflowSpec}
};
use metallic_loader::ModelLoader;
use metallic_sdk::debug::op_metrics_enabled;
use rand;
use rustc_hash::FxHashMap;

fn summarize_op_metrics(results: &FxHashMap<String, Value>) -> Option<String> {
    if !op_metrics_enabled() {
        return None;
    }
    let metrics = results.get("_internal.op_metrics")?.as_map()?;
    let mut entries: Vec<(String, usize, usize, usize)> = Vec::with_capacity(metrics.len());
    for (name, val) in metrics {
        let Some(map) = val.as_map() else {
            continue;
        };
        let total_us = map.get("total_us").and_then(Value::as_usize).unwrap_or(0);
        let count = map.get("count").and_then(Value::as_usize).unwrap_or(0);
        let max_us = map.get("max_us").and_then(Value::as_usize).unwrap_or(0);
        entries.push((name.clone(), total_us, count, max_us));
    }
    if entries.is_empty() {
        return None;
    }
    entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top = entries
        .into_iter()
        .take(6)
        .map(|(name, total_us, count, max_us)| {
            format!(
                "{} total={:.2}ms count={} max={:.2}ms",
                name,
                total_us as f64 / 1000.0,
                count,
                max_us as f64 / 1000.0
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    Some(top)
}

/// Information about an available model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Display name for the UI (derived from filename).
    pub display_name: String,
    /// Full path to the GGUF file.
    pub path: PathBuf,
}

impl ModelInfo {
    /// Create from a path, extracting the display name from the filename.
    pub fn from_path(path: PathBuf) -> Option<Self> {
        let display_name = path.file_stem()?.to_string_lossy().into_owned();
        Some(Self { display_name, path })
    }
}

/// Status of model loading operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ModelLoadStatus {
    /// No model loading in progress.
    #[default]
    Idle,
    /// Model is currently being loaded.
    Loading,
    /// Model loaded successfully.
    Loaded,
    /// Model loading failed with error message.
    Error(String),
}

/// Scan the models directory for available GGUF files.
///
/// Looks for `.gguf` files in `./models/` relative to the current working directory.
pub fn scan_models_directory() -> Vec<ModelInfo> {
    let models_dir = std::env::current_dir()
        .map(|cwd| cwd.join("models"))
        .unwrap_or_else(|_| PathBuf::from("models"));

    let Ok(entries) = std::fs::read_dir(&models_dir) else {
        tracing::warn!("Could not read models directory: {:?}", models_dir);
        return Vec::new();
    };

    let mut models: Vec<ModelInfo> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            // Only include .gguf files
            if path.extension().is_some_and(|ext| ext == "gguf") && path.is_file() {
                ModelInfo::from_path(path)
            } else {
                None
            }
        })
        .collect();

    // Sort by display name for consistent ordering
    models.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    tracing::info!("Found {} GGUF models in {:?}", models.len(), models_dir);
    models
}

/// The state of a loaded model, ready for inference.
#[derive(Clone)]
pub struct LoadedModelState {
    /// The Foundry instance (owns GPU resources), wrapped for shared thread-safe access.
    pub foundry: Arc<Mutex<Foundry>>,
    /// The compiled model (Arc-wrapped, thread-safe via internal locks).
    pub model: Arc<CompiledModel>,
    /// Full path to the GGUF file used.
    pub path: PathBuf,
    /// The models map for WorkflowRunner.
    pub models: FxHashMap<String, Arc<CompiledModel>>,
    /// Persistent workflow runner preserving per-op state across requests.
    pub runner: Arc<Mutex<WorkflowRunner>>,
    /// The multi-turn chat workflow.
    pub workflow: WorkflowSpec,
}

/// Result of model loading operation.
pub type LoadModelResult = Result<LoadedModelState, String>;

fn load_workflow(workflow_path: &Path) -> Result<WorkflowSpec, String> {
    tracing::debug!("Loading workflow from {:?}", workflow_path);

    let file = std::fs::File::open(workflow_path).map_err(|e| format!("Failed to open workflow file {:?}: {}", workflow_path, e))?;
    serde_json::from_reader(file).map_err(|e| format!("Failed to parse workflow: {}", e))
}

/// Load a model from a GGUF file path.
///
/// If `system_prompt` is provided, a "warmup" inference pass is performed to pre-fill
/// the KV cache and compile Metal kernels.
pub fn load_model(gguf_path: &Path, system_prompt: Option<String>) -> LoadModelResult {
    tracing::info!("Loading model from {:?}", gguf_path);

    // 1. Load the model from the GGUF file (to get architecture)
    tracing::info!("Loading GGUF model...");
    let model_loaded = ModelLoader::from_file(gguf_path).map_err(|e| format!("Failed to load model: {}", e))?;

    // 2. Resolve architecture routing (spec + workflow) from shared registry.
    let routing = resolve_model_routing_from_loaded_model(model_loaded.as_ref())?;
    tracing::info!("Detected architecture: {} (rule={})", routing.architecture, routing.matched_rule);
    tracing::debug!("Using spec file: {:?}", routing.spec_path);
    tracing::debug!("Using workflow file: {:?}", routing.workflow_path);

    // 4. Initialize Foundry
    tracing::info!("Initializing Foundry...");
    let foundry = Foundry::new().map_err(|e| format!("Failed to initialize Foundry: {}", e))?;

    // 5. Build the compiled model
    tracing::info!("Building compiled model...");
    let model = ModelBuilder::new()
        .with_spec_file(routing.spec_path.clone())
        .map_err(|e| format!("Failed to load spec file: {}", e))?
        .with_model(model_loaded)
        .build_lazy()
        .map_err(|e| format!("Failed to build model: {}", e))?;

    let model = Arc::new(model);

    // 6. Create models map for WorkflowRunner
    let mut models = FxHashMap::default();
    models.insert("llm".to_string(), Arc::clone(&model));

    // 7. Load the routed workflow.
    let workflow = load_workflow(&routing.workflow_path)?;

    tracing::info!("Model loaded successfully");

    let state = LoadedModelState {
        foundry: Arc::new(Mutex::new(foundry)),
        model,
        path: gguf_path.to_path_buf(),
        runner: Arc::new(Mutex::new(WorkflowRunner::new(models.clone()))),
        models,
        workflow,
    };

    // 8. Optional Warmup / System Prefill
    if let Some(sys_prompt) = system_prompt {
        tracing::info!("Performing model warmup with system prompt...");
        let mut inputs = FxHashMap::default();
        inputs.insert("system_prompt".to_string(), Value::Text(Arc::from(sys_prompt.as_str())));
        // Warm both prefill and decode/sample kernels without producing long output.
        inputs.insert("run_generation".to_string(), Value::U32(1));
        inputs.insert("max_tokens".to_string(), Value::U32(1));
        // Include an empty user turn so warmup stores a reusable "system+user scaffold" KV prefix.
        let mut warmup_user = FxHashMap::default();
        warmup_user.insert("role".to_string(), Value::Text(Arc::from("user")));
        warmup_user.insert("content".to_string(), Value::Text(Arc::from("")));
        inputs.insert("messages".to_string(), Value::Array(vec![Value::Map(warmup_user)]));
        // Workflow output placeholder required by the runner schema.
        inputs.insert("generated_tokens".to_string(), Value::TokensU32(vec![]));

        let (tx, _rx) = smol::channel::bounded(1);
        let cancel = AtomicBool::new(false);

        // Run warmup pass in the background of this load task.
        // This primes the "system+empty-user" KV scaffold and warms up Metal kernels.
        let _ = run_inference_streaming(&state, inputs, tx, &cancel).map_err(|e| format!("Model warmup failed: {}", e))?;

        // Do not leak warmup context into real chat turns. Keep allocations/bindings hot.
        state.model.rewind_session();
        if let Ok(mut runner) = state.runner.lock() {
            runner.reset();
        }

        tracing::info!("Model warmup/prefill completed");
    }

    Ok(state)
}

/// Convert a role string to the format expected by the workflow.
pub(crate) fn role_to_string(role: crate::types::MessageRole) -> &'static str {
    match role {
        crate::types::MessageRole::User => "user",
        crate::types::MessageRole::Assistant => "assistant",
        crate::types::MessageRole::System => "system",
    }
}

pub(crate) fn messages_to_value(messages: &[crate::types::ChatMessage]) -> Value {
    let messages_vec: Vec<Value> = messages
        .iter()
        .map(|msg| {
            let mut map = FxHashMap::default();
            map.insert("role".to_string(), Value::Text(Arc::from(role_to_string(msg.role))));
            map.insert("content".to_string(), Value::Text(Arc::from(msg.content.as_str())));
            Value::Map(map)
        })
        .collect();
    Value::Array(messages_vec)
}

/// Run inference on a list of chat messages and return the generated response.
///
/// This version is blocking and should be wrapped in smol::unblock.
pub fn run_inference(loaded_model: &LoadedModelState, messages: &[crate::types::ChatMessage]) -> Result<String> {
    let (tx, rx) = smol::channel::bounded(32);
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Clone data for the background thread
    let loaded_model_clone = loaded_model.clone();
    let messages_vec = messages.to_vec();

    std::thread::scope(|s| {
        let runner_handle = s.spawn(move || {
            let cancel = AtomicBool::new(false);
            let mut inputs = FxHashMap::default();
            let messages_value = messages_to_value(&messages_vec);
            inputs.insert("messages".to_string(), messages_value);
            run_inference_streaming(&loaded_model_clone, inputs, tx, &cancel)
        });

        while let Ok(token_id) = rx.recv_blocking() {
            generated_tokens.push(token_id);
        }

        match runner_handle.join() {
            Ok(result) => result,
            Err(panic_payload) => {
                let panic_msg = panic_payload
                    .downcast_ref::<&str>()
                    .map(|msg| (*msg).to_string())
                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic payload".to_string());
                Err(anyhow::anyhow!("Inference worker panicked: {panic_msg}"))
            }
        }
    })?;

    tracing::info!("Finished generating {} tokens", generated_tokens.len());

    // Decode tokens to text using the model's tokenizer
    let tokenizer = loaded_model.model.tokenizer().map_err(|e| anyhow::anyhow!(e))?;
    let text = tokenizer.decode(&generated_tokens).map_err(|e| anyhow::anyhow!(e))?;
    Ok(text)
}

/// Run inference and stream token IDs through a channel.
///
/// This is the core inference function designed to run in a background thread.
pub fn run_inference_streaming(
    loaded_model: &LoadedModelState,
    inputs: FxHashMap<String, Value>,
    token_tx: smol::channel::Sender<u32>,
    cancel: &AtomicBool,
) -> Result<crate::types::InferencePerf> {
    tracing::info!("Running streaming inference with {} inputs", inputs.len());

    let mut inputs = inputs;

    let max_tokens_cap = std::env::var("METALLIC_GUI_MAX_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0);
    if let Some(cap) = max_tokens_cap {
        inputs.insert("max_tokens".to_string(), Value::Usize(cap));
    }

    // Handle random seed
    if let Some(Value::Bool(true)) = inputs.get("seed_random") {
        inputs.insert("seed".to_string(), Value::U32(rand::random()));
    }

    // Lock foundry for the duration of this inference call
    let mut foundry_guard = loaded_model.foundry.lock().map_err(|_| anyhow::anyhow!("Foundry lock poisoned"))?;
    let mut runner_guard = loaded_model
        .runner
        .lock()
        .map_err(|_| anyhow::anyhow!("WorkflowRunner lock poisoned"))?;
    let generation_started = std::time::Instant::now();
    let mut emitted_tokens: usize = 0;
    let repeat_token_hard_stop = std::env::var("METALLIC_REPEAT_TOKEN_HARD_STOP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut last_token_id: Option<u32> = None;
    let mut same_token_run_len: usize = 0;
    let mut prefill_us: Option<u128> = None;
    let mut setup_us: Option<u128> = None;
    let mut decode_us_total: u128 = 0;
    let mut decode_samples: usize = 0;
    let mut first_token_latency_us: Option<u128> = None;
    let mut first_decode_us: Option<u128> = None;

    let results = runner_guard
        .run_streaming(
            &mut foundry_guard,
            &loaded_model.workflow,
            inputs,
            |token_id, prefill_dur, setup_dur, decode_dur| {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }

                if repeat_token_hard_stop > 0 {
                    if last_token_id == Some(token_id) {
                        same_token_run_len = same_token_run_len.saturating_add(1);
                    } else {
                        last_token_id = Some(token_id);
                        same_token_run_len = 1;
                    }
                }

                emitted_tokens += 1;
                if prefill_us.is_none() {
                    prefill_us = Some(prefill_dur.as_micros());
                }
                if setup_us.is_none() {
                    setup_us = Some(setup_dur.as_micros());
                }
                if let Some(decode_dur) = decode_dur {
                    decode_us_total = decode_us_total.saturating_add(decode_dur.as_micros());
                    decode_samples = decode_samples.saturating_add(1);
                    if first_decode_us.is_none() {
                        first_decode_us = Some(decode_dur.as_micros());
                    }
                }
                if first_token_latency_us.is_none() {
                    first_token_latency_us = Some(generation_started.elapsed().as_micros());
                }
                if token_tx.send_blocking(token_id).is_err() {
                    return Ok(false); // Stop if receiver is gone
                }
                if repeat_token_hard_stop > 0 && same_token_run_len >= repeat_token_hard_stop {
                    tracing::warn!(
                        "Stopping generation due to repeated-token loop: token_id={} repeated {} times",
                        token_id,
                        same_token_run_len
                    );
                    return Ok(false);
                }
                Ok(true)
            },
        )
        .map_err(|e| anyhow::anyhow!(e))?;

    let elapsed = generation_started.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let wall_tok_per_sec = if elapsed_secs > 0.0 {
        emitted_tokens as f64 / elapsed_secs
    } else {
        0.0
    };
    let decode_tok_per_sec = if decode_us_total > 0 {
        (decode_samples as f64) / (decode_us_total as f64 / 1_000_000.0)
    } else {
        let first_token_latency = first_token_latency_us.unwrap_or(0) as f64 / 1_000_000.0;
        let decode_window_secs = (elapsed_secs - first_token_latency).max(0.0);
        if decode_window_secs > 0.0 {
            emitted_tokens as f64 / decode_window_secs
        } else {
            0.0
        }
    };

    let prefill_tokens = results.get("_internal.prompt_len").and_then(|v| v.as_usize()).unwrap_or(0);
    let context_tokens = results
        .get("_internal.start_pos")
        .and_then(|v| v.as_usize())
        .and_then(|start_pos| start_pos.checked_add(prefill_tokens))
        .and_then(|prefill_end| prefill_end.checked_add(emitted_tokens));
    let total_prefill_us = results.get("_internal.prefill_us").and_then(|v| v.as_usize()).unwrap_or(0) as f64;
    let prefill_tok_per_sec = if total_prefill_us > 0.0 {
        (prefill_tokens as f64) / (total_prefill_us / 1_000_000.0)
    } else {
        0.0
    };

    let first_token_ms = first_token_latency_us.unwrap_or(0) / 1_000;
    let prefill_ms = prefill_us.unwrap_or(0) / 1_000;
    let setup_ms = setup_us.unwrap_or(0) / 1_000;
    let first_decode_ms = first_decode_us.unwrap_or(0) / 1_000;
    let decode_wait_ms = first_token_ms.saturating_sub(prefill_ms.saturating_add(setup_ms));
    let prompt_prep_ms = decode_wait_ms.saturating_sub(first_decode_ms);
    let include_breakdown = op_metrics_enabled();
    let hit_max_tokens_cap = max_tokens_cap.is_some_and(|cap| emitted_tokens >= cap);
    if let Some(op_metrics_summary) = summarize_op_metrics(&results) {
        tracing::debug!("Inference op metrics: {}", op_metrics_summary);
    }
    if include_breakdown {
        tracing::debug!(
            "Inference metrics: tokens={}, prefill_tokens={}, max_tokens_cap={}, hit_max_tokens_cap={}, wall_elapsed_ms={}, wall_tok_per_sec={:.2}, decode_tok_per_sec={:.2}, prefill_tok_per_sec={:.2}, first_token_latency_ms={}, prefill_ms={}, setup_ms={}, prompt_prep_ms={}, first_decode_ms={}, decode_wait_ms={}, decode_samples={}",
            emitted_tokens,
            prefill_tokens,
            max_tokens_cap.unwrap_or(0),
            hit_max_tokens_cap,
            elapsed.as_millis(),
            wall_tok_per_sec,
            decode_tok_per_sec,
            prefill_tok_per_sec,
            first_token_ms,
            prefill_ms,
            setup_ms,
            prompt_prep_ms,
            first_decode_ms,
            decode_wait_ms,
            decode_samples
        );
    } else {
        tracing::debug!(
            "Inference metrics: tokens={}, prefill_tokens={}, max_tokens_cap={}, hit_max_tokens_cap={}, wall_elapsed_ms={}, wall_tok_per_sec={:.2}, decode_tok_per_sec={:.2}, prefill_tok_per_sec={:.2}, first_token_latency_ms={}, prefill_ms={}, decode_samples={}",
            emitted_tokens,
            prefill_tokens,
            max_tokens_cap.unwrap_or(0),
            hit_max_tokens_cap,
            elapsed.as_millis(),
            wall_tok_per_sec,
            decode_tok_per_sec,
            prefill_tok_per_sec,
            first_token_ms,
            prefill_ms,
            decode_samples
        );
    }

    Ok(crate::types::InferencePerf {
        tokens: emitted_tokens,
        wall_ms: elapsed.as_millis(),
        wall_tok_per_sec,
        decode_tok_per_sec,
        first_token_ms,
        prefill_ms,
        setup_ms: include_breakdown.then_some(setup_ms),
        prompt_prep_ms: include_breakdown.then_some(prompt_prep_ms),
        first_decode_ms: include_breakdown.then_some(first_decode_ms),
        decode_wait_ms: include_breakdown.then_some(decode_wait_ms),
        prefill_tokens,
        prefill_tok_per_sec,
        context_tokens,
    })
}
