use std::{any::Any, io::Write, sync::mpsc, time::Duration};

use anyhow::Result;
use metallic_cli_helpers::app_event::AppEvent;

use crate::{cli, tui::App};

pub(crate) fn emit_startup_memory_update(tx: &mpsc::Sender<AppEvent>) -> Result<(), mpsc::SendError<AppEvent>> {
    // Send empty memory rows as placeholder in the new system
    tx.send(AppEvent::MemoryUpdate(vec![]))
}

pub(crate) fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "unknown panic".to_string()
    }
}

pub(crate) fn handle_app_event(app: &mut App, event: AppEvent) {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_text_mode_event(
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
