use std::{
    any::Any, io::{Write, stdout}, panic::{self, AssertUnwindSafe}, sync::mpsc, thread, time::Duration
};

use anyhow::Result;
use metallic::{
    Context, F16Element, TensorElement, Tokenizer, generation::generate_streaming, gguf::{GGUFFile, model_loader::GGUFModelLoader}, kernels::{KernelBackendKind, KernelBackendOverride, KernelBackendOverrides}, profiling_state
};
use metallic_cli_helpers::prelude::*;
use metallic_instrumentation::{MetricEvent, config::AppConfig, prelude::*, record_metric_async};
use rustc_hash::FxHashMap;

mod cli;
mod tui;

use clap::Parser;
use crossterm::event::{self, Event as CrosstermEvent, MouseButton, MouseEvent};
use ratatui::{
    Terminal, backend::{Backend, CrosstermBackend}, layout::Position
};
use tui::{App, AppResult, app::FocusArea, ui};

fn main() -> AppResult<()> {
    // Ensure profiling state is initialized from the environment before anything else.
    // This avoids a race condition where the TUI might read the state before the
    // generation thread initializes it.
    profiling_state::initialize_profiling_state_from_env();

    // Parse command line arguments using CLAP
    let cli_config = cli::CliConfig::parse();

    // Load instrumentation config from the environment so exporter selection honours CLI env vars.
    let app_config = AppConfig::get_or_init_from_env().map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;

    // Initialize instrumentation system with async recorder for zero-overhead metrics
    let (sender, receiver) = mpsc::channel();
    let channel_exporter = Box::new(ChannelExporter::new(sender));

    let mut exporters: Vec<Box<dyn MetricExporter>> = vec![channel_exporter];

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

                // Report GGUF file MMAP usage
                let gguf_file_size = std::fs::metadata(&gguf_path)?.len();
                record_metric_async!(MetricEvent::GgufFileMmap {
                    size_bytes: gguf_file_size
                });

                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
                let mut ctx = Context::<F16Element>::new()?;

                // Apply global backend override first (affects all kernels that consult the registry)
                if let Some(choice) = cli_config.backend {
                    let override_policy = match choice {
                        cli::config::GlobalBackendChoice::Auto => KernelBackendOverride::Auto,
                        cli::config::GlobalBackendChoice::Legacy => KernelBackendOverride::Force(KernelBackendKind::Legacy),
                        cli::config::GlobalBackendChoice::Graph => KernelBackendOverride::Force(KernelBackendKind::Graph),
                    };
                    ctx.set_global_backend_override(override_policy);
                }

                // Then apply per-op SDPA override if provided (takes precedence for sdpa)
                if let Some(choice) = cli_config.sdpa_backend {
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

                worker_tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
                let loader = GGUFModelLoader::new(gguf);
                emit_startup_memory_update(&worker_tx)?;
                let gguf_model = loader.load_model()?;
                emit_startup_memory_update(&worker_tx)?;

                worker_tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
                let mut qwen: metallic::models::qwen25::Qwen25<F16Element> = gguf_model.instantiate(&mut ctx)?;

                // Report model weights breakdown
                report_model_weight_breakdown(&qwen);
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
                    kv_initial_headroom_tokens: (cli_config.generation.max_tokens / 4).max(32),
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

fn report_model_weight_breakdown(qwen: &metallic::models::Qwen25<F16Element>) {
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
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();

    // Get the initial profiling state and set it in the app
    let initial_profiling_state = profiling_state::get_profiling_state();
    app.set_profiling_active(initial_profiling_state);

    while !app.should_quit {
        // Handle crossterm events
        if crossterm::event::poll(std::time::Duration::from_millis(50))? {
            match crossterm::event::read()? {
                CrosstermEvent::Key(key) => {
                    if key.code == crossterm::event::KeyCode::Char('p') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                        // Toggle the state, then get the new state
                        profiling_state::toggle_profiling_state();
                        let new_state = profiling_state::get_profiling_state();
                        app.set_profiling_active(new_state);
                    } else {
                        match key.code {
                            crossterm::event::KeyCode::Char('q') => app.quit(),
                            crossterm::event::KeyCode::Enter | crossterm::event::KeyCode::Char(' ') | crossterm::event::KeyCode::Esc => {
                                if app.has_active_alert() {
                                    app.dismiss_active_alert();
                                }
                            }
                            crossterm::event::KeyCode::Char('m') => {
                                app.metrics_view = tui::app::MetricsView::Memory;
                                app.reset_metrics_scroll();
                            }
                            crossterm::event::KeyCode::Char('l') => {
                                // Check if it's Control+L for log toggle
                                if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                    app.toggle_log_visibility();
                                } else {
                                    // Regular 'l' key to switch to latency view
                                    app.metrics_view = tui::app::MetricsView::Latency;
                                    app.reset_metrics_scroll();
                                }
                            }
                            crossterm::event::KeyCode::Char('s') => {
                                app.metrics_view = tui::app::MetricsView::Stats;
                                app.reset_metrics_scroll();
                            }
                            crossterm::event::KeyCode::Char('c') => {
                                // Check if it's Control+C for copying all content from focused widget
                                if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                    // Copy all content from the currently focused widget
                                    let content_to_copy = match app.focus {
                                        FocusArea::GeneratedText => app.generated_text.clone(),
                                        FocusArea::Metrics => {
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
                                    };
                                    copy_text_to_clipboard(&content_to_copy);
                                } else {
                                    app.toggle_collapse();
                                }
                            }
                            // Add new key binding for toggling between markdown and plain text display
                            crossterm::event::KeyCode::Tab => {
                                app.focus_next();
                                app.reset_metrics_scroll();
                            }
                            crossterm::event::KeyCode::Char('d') => {
                                if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                    // Toggle between markdown and plain text display
                                    app.text_display_mode = match app.text_display_mode {
                                        tui::app::TextDisplayMode::Plain => tui::app::TextDisplayMode::Markdown,
                                        tui::app::TextDisplayMode::Markdown => tui::app::TextDisplayMode::Plain,
                                    };
                                } else {
                                    // For regular 'd' key, could add some other functionality if wanted
                                    // For now, just handle as a normal character key - but since it's not mapped to anything, it's ignored
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
    _receiver: &std::sync::mpsc::Receiver<EnrichedMetricEvent>,
    rx: &std::sync::mpsc::Receiver<AppEvent>,
    generation_handle: thread::JoinHandle<Result<()>>,
) -> AppResult<()> {
    use tracing::{debug, error, info, warn};

    let mut generated_tokens: u64 = 0;
    let mut prompt_processing: Option<Duration> = None;
    let mut total_generation_time: Option<Duration> = None;

    // Process all events until generation is done
    while !generation_handle.is_finished() || rx.recv_timeout(Duration::from_millis(100)).is_ok() {
        // Process any pending app events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token {
                    text,
                    prompt_processing: prompt,
                    ..
                } => {
                    print!("{}", text);
                    std::io::stdout().flush().unwrap();
                    generated_tokens = generated_tokens.saturating_add(1);
                    prompt_processing.get_or_insert(prompt);
                }
                AppEvent::GenerationComplete {
                    total_generation_time: total,
                } => {
                    total_generation_time = Some(total);
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

    if let Some(total) = total_generation_time {
        let total_s = total.as_secs_f64().max(1e-9);
        let tps_total = (generated_tokens as f64) / total_s;
        eprintln!("\n\n[metallic] generated_tokens={generated_tokens} total_s={total_s:.3} tps_total={tps_total:.2}");

        if let Some(prompt) = prompt_processing {
            let decode_s = total.saturating_sub(prompt).as_secs_f64().max(1e-9);
            let tps_decode = (generated_tokens as f64) / decode_s;
            eprintln!(
                "[metallic] prompt_s={:.3} decode_s={decode_s:.3} tps_decode={tps_decode:.2}",
                prompt.as_secs_f64()
            );
        }
    }

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
                AppEvent::LogMessage(message) => {
                    let log_entry = serde_json::json!({
                        "type": "log_message",
                        "message": message,
                        "timestamp": chrono::Utc::now().to_rfc3339()
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
