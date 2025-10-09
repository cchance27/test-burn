use anyhow::Result;
use metallic::{
    Context, F16Element, Tokenizer,
    generation::generate_streaming,
    gguf::{GGUFFile, model_loader::GGUFModelLoader},
};
use metallic_cli_helpers::prelude::*;
use metallic_instrumentation::prelude::*;
use std::{
    any::Any,
    io::{Write, stdout},
    panic::{self, AssertUnwindSafe},
    sync::mpsc,
    thread,
    time::Duration,
};

mod cli;
mod tui;

use clap::Parser;
use ratatui::{
    Terminal,
    backend::{Backend, CrosstermBackend},
};
use tui::{App, AppResult};

const GENERATION_LOOP_LABEL: &str = "Generation Loop";
const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

fn main() -> AppResult<()> {
    // Parse command line arguments using CLAP
    let cli_config = cli::CliConfig::parse();

    // Set up tracing based on verbosity level and output format
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt};

    // In JSON mode, only show errors by default to avoid interfering with JSON output
    let filter = if matches!(cli_config.output_format, cli::config::OutputFormat::Json) {
        match cli_config.verbose {
            0 => EnvFilter::new("metallic=error,metallic_cli=error,tui=error,metallic_instrumentation=error,metrics=info"),
            1 => EnvFilter::new("metallic=warn,metallic_cli=warn,tui=warn,metallic_instrumentation=warn,metrics=info"),
            2 => EnvFilter::new("metallic=info,metallic_cli=info,tui=info,metallic_instrumentation=info,metrics=info"),
            _ => EnvFilter::new("metallic=debug,metallic_cli=debug,tui=debug,metallic_instrumentation=debug,metrics=info"),
        }
    } else {
        // For TUI and text modes, use normal verbosity levels
        match cli_config.verbose {
            0 => EnvFilter::new("metallic=warn,metallic_cli=warn,tui=warn,metallic_instrumentation=warn,metrics=info"),
            1 => EnvFilter::new("metallic=info,metallic_cli=info,tui=info,metallic_instrumentation=info,metrics=info"),
            2 => EnvFilter::new("metallic=debug,metallic_cli=debug,tui=debug,metallic_instrumentation=debug,metrics=info"),
            _ => EnvFilter::new("metallic=trace,metallic_cli=trace,tui=trace,metallic_instrumentation=trace,metrics=info"),
        }
    };

    // Initialize instrumentation system with tracing subscriber and metrics layer
    let (sender, receiver) = mpsc::channel();
    let channel_exporter = Box::new(ChannelExporter::new(sender));
    //let console_exporter = Box::new(ConsoleExporter::new());

    let exporters: Vec<Box<dyn MetricExporter>> = vec![channel_exporter]; //console_exporter];

    let metrics_layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(filter).with(metrics_layer);

    tracing::subscriber::set_global_default(subscriber).expect("setting global default subscriber failed");

    alert::init_error_logging();
    let gguf_path = cli_config.gguf_path.clone();
    let prompt = cli_config.get_prompt();

    let (tx, rx) = mpsc::channel();

    let generation_handle = {
        let worker_tx = tx.clone();
        thread::spawn(move || -> Result<()> {
            let worker = || -> Result<()> {
                // Send initial empty memory update - simplified for new system
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Loading GGUF Metadata...".to_string()))?;
                let gguf = GGUFFile::load_mmap_and_get_metadata(&gguf_path)?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
                let mut ctx = Context::<F16Element>::new()?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
                let loader = GGUFModelLoader::new(gguf);
                emit_startup_memory_update(&worker_tx)?;
                let gguf_model = loader.load_model()?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
                let mut qwen = gguf_model.instantiate(&mut ctx)?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing tokenizer...".to_string()))?;
                let tokenizer = Tokenizer::from_gguf_metadata(&gguf_model.metadata)?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
                let tokens = tokenizer.encode(&prompt)?;
                worker_tx.send(AppEvent::TokenCount(tokens.len()))?;
                emit_startup_memory_update(&worker_tx)?;

                let cfg = metallic::generation::GenerationConfig {
                    max_tokens: cli_config.generation.max_tokens,
                    temperature: cli_config.generation.temperature as f32,
                    top_p: cli_config.generation.top_p as f32,
                    top_k: cli_config.generation.top_k,
                };

                worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
                generate_streaming(&mut qwen, &tokenizer, &mut ctx, &prompt, &cfg, &worker_tx)?;
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
        cli::config::OutputFormat::Tui => run_tui_mode(&receiver, &rx, generation_handle)?,
        cli::config::OutputFormat::Text => run_text_mode(&receiver, &rx, generation_handle)?,
        cli::config::OutputFormat::Json => run_json_mode(&receiver, &rx, generation_handle)?,
    }

    Ok(())
}

fn run_tui_mode(
    receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new(cli::config::GenerationConfig::default());

    while !app.should_quit {
        // Handle crossterm events
        if crossterm::event::poll(std::time::Duration::from_millis(50))?
            && let crossterm::event::Event::Key(key) = crossterm::event::read()?
        {
            match key.code {
                crossterm::event::KeyCode::Char('q') => app.should_quit = true,
                crossterm::event::KeyCode::Char('m') => {
                    app.metrics_view = tui::app::MetricsView::Memory;
                    app.reset_metrics_scroll();
                }
                crossterm::event::KeyCode::Char('l') => {
                    app.metrics_view = tui::app::MetricsView::Latency;
                    app.reset_metrics_scroll();
                }
                crossterm::event::KeyCode::Char('c') => {
                    app.toggle_collapse();
                }
                crossterm::event::KeyCode::Tab => {
                    app.focus_next();
                    app.reset_metrics_scroll();
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

        // Process metric events from the instrumentation system and convert them to AppEvents
        while let Ok(enriched_event) = receiver.try_recv() {
            let rows = metric_event_to_latency_rows(&enriched_event.event);
            if !rows.is_empty() {
                // Send the rows as an AppEvent so they get processed like before
                handle_app_event(&mut app, AppEvent::LatencyUpdate(rows));
            }
        }

        // Process app events
        while let Ok(event) = rx.try_recv() {
            handle_app_event(&mut app, event);
        }

        terminal.draw(|frame| tui::ui::render(&mut app, frame))?;
    }

    restore_terminal()?;
    generation_handle.join().unwrap()?;
    Ok(())
}

fn run_text_mode(
    _receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    use tracing::{debug, error, info, warn};

    // Process all events until generation is done
    while !generation_handle.is_finished() || rx.recv_timeout(Duration::from_millis(100)).is_ok() {
        // Process any pending app events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token { text, .. } => {
                    print!("{}", text);
                    std::io::stdout().flush().unwrap();
                }
                AppEvent::StatusUpdate(status) => {
                    // Only log status updates if we're in verbose mode
                    // For now, using debug level which won't show by default
                    debug!("Status update: {}", status);
                }
                AppEvent::Alert(alert) => match alert.level {
                    metallic_cli_helpers::app_event::AlertLevel::Info => {
                        info!("{}", alert.message);
                    }
                    metallic_cli_helpers::app_event::AlertLevel::Warning => {
                        warn!("{}", alert.message);
                    }
                    metallic_cli_helpers::app_event::AlertLevel::Error => {
                        error!("{}", alert.message);
                    }
                },
                AppEvent::TokenCount(count) => {
                    debug!("Prompt token count: {}", count);
                }
                _ => {} // Ignore other events in text mode
            }
        }
    }

    generation_handle.join().unwrap()?;
    Ok(())
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
        }
        AppEvent::StatusUpdate(status) => {
            app.status = status;
        }
        AppEvent::MemoryUpdate(memory_rows) => {
            app.memory_rows = memory_rows;
            // Recalculate max depth for memory metrics since the rows may have changed
            app.memory_collapse_depth.calculate_max_depth(&app.memory_rows);
        }
        AppEvent::LatencyUpdate(new_rows) => {
            // Process the incoming metrics and add them to our hierarchical structure
            for row in new_rows {
                app.add_latency_metric(row);
            }
        }
        AppEvent::Alert(alert) => {
            app.push_alert(alert);
        }
    }
}

const BLOCK_STAGE_PREFIXES: &[(&str, &str)] = &[
    ("attn_residual_clone_block_", "attn_residual_clone"),
    ("attn_norm_block_", "attn_norm"),
    ("qkv_proj_block_", "attn_qkv_proj"),
    ("attn_rearrange_block_", "attn_rearrange"),
    ("rope_block_", "Rope"),
    ("kv_cache_block_", "kv_cache"),
    ("kv_repeat_block_", "kv_repeat"),
    ("sdpa_block_", "Sdpa"),
    ("attn_reassembly_block_", "attn_reassembly"),
    ("attn_output_block_", "attn_output"),
    ("attn_residual_block_", "attn_residual"),
    ("mlp_residual_clone_block_", "mlp_residual_clone"),
    ("mlp_norm_block_", "mlp_norm"),
    ("mlp_swiglu_block_", "mlp_swiglu"),
    ("mlp_reshape_block_", "mlp_reshape"),
    ("mlp_residual_block_", "mlp_residual"),
    ("mlp_output_block_", "mlp_output"),
];

fn metric_event_to_latency_rows(event: &MetricEvent) -> Vec<metallic_cli_helpers::app_event::LatencyRow> {
    match event {
        MetricEvent::GpuOpCompleted { op_name, duration_us, .. } => map_gpu_op_completed(op_name)
            .into_iter()
            .map(|segments| build_latency_row(segments, *duration_us))
            .collect(),
        MetricEvent::InternalKernelCompleted {
            parent_op_name,
            internal_kernel_name,
            duration_us,
        } => map_internal_kernel(parent_op_name, internal_kernel_name)
            .into_iter()
            .map(|segments| build_latency_row(segments, *duration_us))
            .collect(),
        _ => Vec::new(),
    }
}

fn map_internal_kernel(parent: &str, kernel: &str) -> Option<Vec<String>> {
    let base_parent = parent.to_ascii_lowercase();
    match base_parent.as_str() {
        "sampling" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Sampling".to_string()]),
        "decoding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Decode".to_string()]),
        "generation_loop" => match kernel {
            k if k.starts_with("block_") && k.ends_with("_total") => {
                let idx_str = k.strip_prefix("block_").unwrap().strip_suffix("_total").unwrap();
                let idx = idx_str.parse::<usize>().ok()?;
                Some(vec![
                    GENERATION_LOOP_LABEL.to_string(),
                    "Forward Step".to_string(),
                    format!("Block {}", idx),
                ])
            }
            "pool_reset" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Pool Reset".to_string()]),
            "embedding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Embedding".to_string()]),
            "forward_step_total" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Forward Step".to_string()]),
            "token_push" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Token Push".to_string()]),
            "cache_logging" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Cache Logging".to_string()]),
            "token_callback" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Token Callback".to_string()]),
            "eos_check" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "EOS Check".to_string()]),
            "metric_recording_overhead" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Metric Recording Overhead".to_string()]),
            "logits_sync" => Some(vec![
                GENERATION_LOOP_LABEL.to_string(),
                "Forward Step".to_string(),
                "Logits Sync".to_string(),
            ]),
            "iteration_total" => Some(vec![GENERATION_LOOP_LABEL.to_string()]),
            _ => None,
        },
        "prompt_processing" => match kernel {
            "logits_sync" => Some(vec![PROMPT_PROCESSING_LABEL.to_string(), "Logits Sync".to_string()]),
            _ => None,
        },
        _ => None,
    }
}

fn map_gpu_op_completed(op_name: &str) -> Option<Vec<String>> {
    let base_name = op_name.split('#').next().unwrap_or(op_name);

    // New: handle hierarchical op names produced by Context::with_gpu_scope which uses '/' separators
    if base_name.contains('/')
        && let Some(segments) = map_hierarchical_gpu_op(base_name)
    {
        return Some(segments);
    }

    if let Some(segments) = map_generation_stage(base_name) {
        return Some(segments);
    }
    if let Some(segments) = map_block_stage(base_name) {
        return Some(segments);
    }
    map_prompt_stage(base_name)
}

fn map_hierarchical_gpu_op(path: &str) -> Option<Vec<String>> {
    // Parse the '/'-separated path and reconstruct a canonical hierarchical label path
    // Default to Generation Loop context. Insert Forward Step when we have block/stage context
    // or an inner label that is part of the forward pass. Plain waits (cb_wait/dep_wait) without
    // a block/stage should live directly under Generation Loop.
    let mut has_generation = false;
    let mut block_label: Option<String> = None;
    let mut stage_label: Option<String> = None;
    let mut inner_label: Option<String> = None;

    for seg in path.split('/') {
        if seg.eq_ignore_ascii_case("generation loop") {
            has_generation = true;
            continue;
        }
        if seg.eq_ignore_ascii_case(PROMPT_PROCESSING_LABEL) {
            // Collapse prompt processing into a single line summary
            return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
        }
        if let Some(rest) = seg.strip_prefix("block_")
            && let Ok(idx) = rest.parse::<usize>()
        {
            block_label = Some(format!("Block {}", idx));
            continue;
        }
        if seg.eq_ignore_ascii_case("generation_step_output") {
            inner_label = Some("Output".to_string());
            continue;
        }
        // Explicitly surface wait segments as visible leaves
        if seg.eq_ignore_ascii_case("cb_wait") {
            inner_label = Some("CB Wait".to_string());
            continue;
        }
        if seg.eq_ignore_ascii_case("dep_wait") {
            inner_label = Some("Dep Wait".to_string());
            continue;
        }
        if let Some(stage) = map_block_stage(seg) {
            // Expect [Generation Loop, Forward Step, Block N, Stage]; capture Block and Stage
            if stage.len() >= 3 {
                block_label = Some(stage[2].clone());
            }
            if let Some(last) = stage.last() {
                stage_label = Some(last.clone());
            }
            continue;
        }
        // Map inner op friendly names
        let inner = seg.trim_end_matches("_op");
        let friendly = match inner {
            s if s.starts_with("sdpa_matmul_qk") => Some("QK MatMul"),
            s if s.starts_with("sdpa_softmax") => Some("Softmax"),
            s if s.starts_with("sdpa_matmul_av") => Some("AV MatMul"),
            s if s.starts_with("attn_qkv_proj") => Some("QKV Proj"),
            s if s.starts_with("attn_rearrange") => Some("Rearrange"),
            s if s.starts_with("attn_output") => Some("Attn Output"),
            s if s.starts_with("mlp_swiglu") => Some("SwiGLU"),
            s if s.starts_with("mlp_norm") => Some("MLP Norm"),
            s if s.starts_with("mlp_output") => Some("MLP Output"),
            s if s.starts_with("rope_block") || s == "rope" => Some("Rope"),
            "generation_step_output" => Some("Output"),
            _ => None,
        };
        if let Some(name) = friendly {
            inner_label = Some(name.to_string());
            continue;
        }
    }

    // Build final path
    let mut out: Vec<String> = Vec::new();
    if has_generation {
        out.push(GENERATION_LOOP_LABEL.to_string());
    } else {
        // If not explicitly present, inject it for generation-related scopes
        out.push(GENERATION_LOOP_LABEL.to_string());
    }

    // Decide whether to insert "Forward Step" as an intermediate node.
    let is_plain_wait = matches!(inner_label.as_deref(), Some("CB Wait") | Some("Dep Wait"));
    let has_block_or_stage = block_label.is_some() || stage_label.is_some();
    if has_block_or_stage || !is_plain_wait {
        out.push("Forward Step".to_string());
    }

    if let Some(block) = block_label {
        out.push(block);
    }
    if let Some(stage) = stage_label {
        out.push(stage);
    }
    if let Some(inner) = inner_label {
        out.push(inner);
    }

    if out.is_empty() { None } else { Some(out) }
}

fn map_generation_stage(name: &str) -> Option<Vec<String>> {
    if let Some(rest) = name.strip_prefix("generation_step_")
        && rest.ends_with("_output")
    {
        return Some(vec![GENERATION_LOOP_LABEL.to_string(), "Output".to_string()]);
    }

    if let Some(_idx) = name.strip_prefix("iteration_") {
        return Some(vec![GENERATION_LOOP_LABEL.to_string()]);
    }

    if name == "forward_step" {
        return Some(vec![GENERATION_LOOP_LABEL.to_string(), "Forward Step".to_string()]);
    }

    if let Some(idx_str) = name.strip_prefix("block_")
        && let Ok(idx) = idx_str.parse::<usize>()
    {
        return Some(vec![
            GENERATION_LOOP_LABEL.to_string(),
            "Forward Step".to_string(),
            format!("Block {}", idx),
        ]);
    }

    None
}

fn map_block_stage(name: &str) -> Option<Vec<String>> {
    let base = name.strip_suffix("_op").unwrap_or(name);
    for (prefix, display) in BLOCK_STAGE_PREFIXES {
        if let Some(rest) = base.strip_prefix(prefix)
            && let Ok(idx) = rest.parse::<usize>()
        {
            return Some(vec![
                GENERATION_LOOP_LABEL.to_string(),
                "Forward Step".to_string(),
                format!("Block {}", idx),
                (*display).to_string(),
            ]);
        }
    }
    None
}

fn build_latency_row(segments: Vec<String>, duration_us: u64) -> metallic_cli_helpers::app_event::LatencyRow {
    let level = segments.len().saturating_sub(1) as u8;
    let label = segments.join("::");
    let duration_ms = duration_us as f64 / 1000.0;
    metallic_cli_helpers::app_event::LatencyRow {
        label,
        last_ms: duration_ms,
        average_ms: duration_ms,
        level,
    }
}

fn map_prompt_stage(name: &str) -> Option<Vec<String>> {
    if let Some(rest) = name.strip_prefix("prompt_step_")
        && rest.parse::<usize>().is_ok()
    {
        return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
    }
    None
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
        //crossterm::event::EnableMouseCapture
    )?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal() -> Result<()> {
    crossterm::execute!(
        stdout(),
        crossterm::terminal::LeaveAlternateScreen,
        //crossterm::event::DisableMouseCapture
    )?;
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
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
