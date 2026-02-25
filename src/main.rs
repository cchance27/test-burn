use std::sync::mpsc;

use metallic_cli_helpers::prelude::*;
use metallic_foundry::workflow::Value as WorkflowValue;
use metallic_instrumentation::{config::AppConfig, prelude::*};

mod app;
mod cli;
mod tui;

use std::sync::OnceLock;

use app::{
    modes::{run_json_mode, run_text_mode, run_tui_mode}, workflow_inputs::env_bool
};
use clap::Parser;
use tui::AppResult;

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
    let tui_start_processing =
        matches!(cli_config.output_format, cli::config::OutputFormat::Tui) && prompts.first().is_some_and(|p| !p.trim().is_empty());
    let worker_generation = cli_config.generation;
    let worker_output_format = cli_config.output_format.clone();
    let workflow_path = cli_config.workflow.clone();
    let mut worker_foundry_config = metallic_foundry::FoundryConfig::default();
    for (key, value) in cli_config
        .parsed_foundry_env_overrides()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?
    {
        worker_foundry_config = worker_foundry_config.with_env_override(key, value);
    }
    if let Some(compute_dtype) = cli_config.compute_dtype {
        worker_foundry_config = worker_foundry_config.with_env_override("METALLIC_COMPUTE_DTYPE", compute_dtype.as_env_value());
    }
    if let Some(accum_dtype) = cli_config.accum_dtype {
        worker_foundry_config = worker_foundry_config.with_env_override("METALLIC_ACCUM_DTYPE", accum_dtype.as_env_value());
    }
    let worker_workflow_kwargs = cli_config
        .parsed_workflow_kwargs()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    let kwarg_enable_thinking = worker_workflow_kwargs
        .iter()
        .rev()
        .find(|(key, _)| key == "enable_thinking")
        .and_then(|(_, value)| WorkflowValue::parse_boolish_str(value));
    let worker_thinking_override = cli_config.thinking_override().or(kwarg_enable_thinking);

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

    let generation_handle = app::worker::spawn_generation_worker(app::worker::GenerationWorkerParams {
        tx: tx.clone(),
        cmd_rx,
        gguf_path,
        prompts,
        worker_generation,
        worker_output_format,
        worker_foundry_config,
        workflow_path,
        worker_workflow_kwargs,
        worker_thinking_override,
    });

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
