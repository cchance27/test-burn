use std::{thread, time::Duration};

use anyhow::Result;
use crossterm::event::Event as CrosstermEvent;
use metallic_cli_helpers::prelude::*;
use metallic_instrumentation::prelude::EnrichedMetricEvent;

use super::{
    events::{handle_app_event, process_text_mode_event}, terminal::{copy_text_to_clipboard, handle_mouse_event, restore_terminal, setup_terminal}
};
use crate::{
    cli, tui::{self, App, AppResult, app::FocusArea, ui}
};

pub(crate) fn run_tui_mode(
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
    let initial_profiling_state = metallic_foundry::instrument::get_profiling_state();
    app.set_profiling_active(initial_profiling_state);

    while !app.should_quit {
        // Handle crossterm events
        if crossterm::event::poll(Duration::from_millis(50))? {
            match crossterm::event::read()? {
                CrosstermEvent::Key(key) => {
                    if key.code == crossterm::event::KeyCode::Char('p') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                        // Toggle profiling state
                        metallic_foundry::instrument::toggle_profiling_state();
                        let new_state = metallic_foundry::instrument::get_profiling_state();
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

pub(crate) fn run_text_mode(
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

pub(crate) fn run_json_mode(
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
