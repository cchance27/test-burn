use anyhow::Result;
use chrono::SecondsFormat;
use metallic::{
    Context, F16Element, Tokenizer,
    generation::{GenerationConfig, generate_streaming},
    gguf::{GGUFFile, model_loader::GGUFModelLoader},
};
use metallic_cli_helpers::prelude::*;
use metallic_instrumentation::prelude::*;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use std::{
    any::Any,
    collections::VecDeque,
    env,
    io::stdout,
    panic::{self, AssertUnwindSafe},
    process,
    sync::mpsc,
    thread,
    time::Duration,
};

#[derive(Clone)]
struct HierarchicalMetric {
    label: String,
    last_ms: f64,
    average_ms: f64,
    children: Vec<HierarchicalMetric>,
}

impl HierarchicalMetric {
    fn new(label: String, last_ms: f64, average_ms: f64) -> Self {
        Self {
            label,
            last_ms,
            average_ms,
            children: Vec::new(),
        }
    }

    fn ensure_child(&mut self, label: &str) -> &mut HierarchicalMetric {
        if let Some(position) = self.children.iter().position(|child| child.label == label) {
            &mut self.children[position]
        } else {
            self.children.push(HierarchicalMetric::new(label.to_string(), 0.0, 0.0));
            self.children
                .last_mut()
                .expect("children vector cannot be empty immediately after push")
        }
    }

    fn upsert_path(&mut self, path: &[&str], last_ms: f64, average_ms: f64) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let child = self.ensure_child(label);

        if path.len() == 1 {
            child.last_ms = last_ms;
            child.average_ms = average_ms;
        } else {
            child.upsert_path(&path[1..], last_ms, average_ms);
        }
    }

    // Calculate inclusive timing (the time for this node plus all its descendants)
    // without modifying the stored values
    fn get_inclusive_timing(&self) -> (f64, f64) {
        let mut last_ms_total = self.last_ms;
        let mut average_ms_total = self.average_ms;

        for child in &self.children {
            let (child_last, child_avg) = child.get_inclusive_timing();
            last_ms_total += child_last;
            average_ms_total += child_avg;
        }

        (last_ms_total, average_ms_total)
    }
}

const GENERATION_LOOP_LABEL: &str = "Generation Loop";
const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

fn metric_event_to_latency_rows(event: &MetricEvent) -> Vec<LatencyRow> {
    match event {
        MetricEvent::GpuOpCompleted { op_name, duration_us, .. } => {
            let base_name = op_name.split('#').next().unwrap_or(op_name);
            let segments: Vec<String> = base_name.split('/').map(|s| s.to_string()).collect();
            if segments.is_empty() || (segments.len() == 1 && segments[0].is_empty()) {
                Vec::new()
            } else {
                // Check if this operation is part of prompt processing
                if segments[0] == PROMPT_PROCESSING_LABEL {
                    // For operations under prompt processing, we want to ensure proper hierarchy
                    // but still create the latency row as normal
                    vec![build_latency_row(segments, *duration_us)]
                } else {
                    vec![build_latency_row(segments, *duration_us)]
                }
            }
        }
        MetricEvent::InternalKernelCompleted {
            parent_op_name,
            internal_kernel_name,
            duration_us,
        } => {
            if let Some(segments) = map_internal_kernel(parent_op_name, internal_kernel_name) {
                vec![build_latency_row(segments, *duration_us)]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

fn map_internal_kernel(parent: &str, kernel: &str) -> Option<Vec<String>> {
    let base_parent = parent.to_ascii_lowercase();
    match base_parent.as_str() {
        "sampling" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Sampling".to_string(), kernel.to_string()]),
        "decoding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Decode".to_string(), kernel.to_string()]),
        "generation_loop" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Embedding".to_string(), kernel.to_string()]),
        _ => None,
    }
}

fn build_latency_row(segments: Vec<String>, duration_us: u64) -> LatencyRow {
    let level = segments.len().saturating_sub(1) as u8;
    let label = segments.join("::");
    let duration_ms = duration_us as f64 / 1000.0;
    LatencyRow {
        label,
        last_ms: duration_ms,
        average_ms: duration_ms,
        level,
    }
}

fn main() -> Result<()> {
    // Initialize instrumentation system with tracing subscriber and metrics layer
    let (sender, receiver) = mpsc::channel();
    let channel_exporter = Box::new(ChannelExporter::new(sender));
    //let console_exporter = Box::new(ConsoleExporter::new());

    let exporters: Vec<Box<dyn MetricExporter>> = vec![
        channel_exporter,
        //console_exporter,
    ];

    let metrics_layer = MetricsLayer::new(exporters);
    let subscriber = tracing_subscriber::registry().with(metrics_layer);
    tracing::subscriber::set_global_default(subscriber).expect("setting global default subscriber failed");

    // Minimal CLI:
    //   cargo run -- /path/to/model.gguf [PROMPT]
    let mut args = env::args().skip(1);
    alert::init_error_logging();
    let gguf_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run -- <GGUF_PATH> [PROMPT]");
            process::exit(1);
        }
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "Create a short javascript hello world app.".to_string());

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

                let cfg = GenerationConfig {
                    max_tokens: 4096,
                    temperature: 0.7,
                    top_p: 0.95,
                    top_k: 40,
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

    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::new();

    while !app_state.should_quit {
        if crossterm::event::poll(Duration::from_millis(50))? {
            match crossterm::event::read()? {
                crossterm::event::Event::Key(key) => {
                    if app_state.has_active_alert() {
                        match key.code {
                            crossterm::event::KeyCode::Enter | crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Char(' ') => {
                                app_state.dismiss_active_alert();
                            }
                            crossterm::event::KeyCode::Char('q') => {
                                app_state.should_quit = true;
                            }
                            _ => {}
                        }
                        continue;
                    }

                    match key.code {
                        crossterm::event::KeyCode::Char('q') => app_state.should_quit = true,
                        crossterm::event::KeyCode::Char('m') => {
                            app_state.metrics_view = MetricsView::Memory;
                            app_state.reset_metrics_scroll();
                        }
                        crossterm::event::KeyCode::Char('l') => {
                            app_state.metrics_view = MetricsView::Latency;
                            app_state.reset_metrics_scroll();
                        }
                        crossterm::event::KeyCode::Char('c') => {
                            app_state.toggle_collapse();
                        }
                        crossterm::event::KeyCode::Tab => {
                            if key.modifiers.contains(crossterm::event::KeyModifiers::SHIFT) {
                                app_state.focus_prev();
                            } else {
                                app_state.focus_next();
                            }
                        }
                        crossterm::event::KeyCode::BackTab => {
                            app_state.focus_prev();
                        }
                        crossterm::event::KeyCode::Up => app_state.scroll_active(-1),
                        crossterm::event::KeyCode::Down => app_state.scroll_active(1),
                        crossterm::event::KeyCode::PageUp => app_state.scroll_active(-10),
                        crossterm::event::KeyCode::PageDown => app_state.scroll_active(10),
                        crossterm::event::KeyCode::Home => app_state.scroll_active_to_start(),
                        crossterm::event::KeyCode::End => app_state.scroll_active_to_end(),
                        _ => {}
                    }
                }
                crossterm::event::Event::Mouse(mouse) => {
                    if let crossterm::event::MouseEventKind::Down(_) = mouse.kind {
                        app_state.handle_click(mouse.column, mouse.row);
                    }
                }
                _ => {}
            }
        }

        // Process metric events from the instrumentation system
        while let Ok(event) = receiver.try_recv() {
            let rows = metric_event_to_latency_rows(&event.event);
            if !rows.is_empty() {
                tx.send(AppEvent::LatencyUpdate(rows))?;
            }
        }

        // Process app events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token {
                    text,
                    prompt_processing,
                    iteration,
                } => {
                    let was_following = app_state.text_follow_bottom;
                    app_state.generated_text.push_str(&text);
                    if was_following {
                        app_state.request_follow_text = true;
                    }
                    app_state.update_generation_metrics(iteration);
                    app_state.prompt_processing_time = prompt_processing;
                }
                AppEvent::TokenCount(count) => {
                    app_state.reset_generation_metrics();
                    app_state.reset_prompt_processing_metrics();
                    app_state.prompt_token_count = count;
                }
                AppEvent::StatusUpdate(status) => {
                    app_state.status = status;
                }
                AppEvent::MemoryUpdate(memory_rows) => {
                    app_state.memory_rows = memory_rows;
                    // Recalculate max depth for memory metrics since the rows may have changed
                    app_state.memory_collapse_depth.calculate_max_depth(&app_state.memory_rows);
                }
                AppEvent::LatencyUpdate(new_rows) => {
                    // Process the incoming metrics and add them to our hierarchical structure
                    for row in new_rows {
                        app_state.add_latency_metric(row);
                    }
                }
                AppEvent::Alert(alert) => {
                    app_state.push_alert(alert);
                }
            }
        }

        terminal.draw(|frame| ui(frame, &mut app_state))?;
    }

    restore_terminal()?;
    generation_handle.join().unwrap()?;
    Ok(())
}

const RUNNING_AVERAGE_WINDOW: usize = 10;

#[derive(Clone)]
struct RunningAverage {
    values: [f64; RUNNING_AVERAGE_WINDOW],
    next_index: usize,
    initialized: bool,
}

impl Default for RunningAverage {
    fn default() -> Self {
        Self {
            values: [0.0; RUNNING_AVERAGE_WINDOW],
            next_index: 0,
            initialized: false,
        }
    }
}

impl RunningAverage {
    fn record(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if !self.initialized {
            self.values = [value; RUNNING_AVERAGE_WINDOW];
            self.initialized = true;
            self.next_index = 1 % RUNNING_AVERAGE_WINDOW;
        } else {
            self.values[self.next_index] = value;
            self.next_index = (self.next_index + 1) % RUNNING_AVERAGE_WINDOW;
        }
    }

    fn average(&self) -> f64 {
        if !self.initialized {
            0.0
        } else {
            self.values.iter().sum::<f64>() / RUNNING_AVERAGE_WINDOW as f64
        }
    }

    fn has_samples(&self) -> bool {
        self.initialized
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

struct AppState {
    generated_text: String,
    iteration_latency: RunningAverage,
    prompt_token_count: usize,
    should_quit: bool,
    status: String,
    memory_rows: Vec<MemoryRow>,
    prompt_processing_time: Duration,
    latency_tree: Vec<HierarchicalMetric>,
    metrics_view: MetricsView,
    latency_collapse_depth: CollapseDepth,
    memory_collapse_depth: MemoryCollapseDepth,
    focus: FocusArea,
    text_scroll: u16,
    metrics_scroll: u16,
    text_follow_bottom: bool,
    request_follow_text: bool,
    text_area: Rect,
    metrics_area: Rect,
    alert_queue: VecDeque<Alert>,
    active_alert: Option<Alert>,
    // Track prompt processing metrics separately for aggregation
    prompt_processing_total_last_ms: f64,
    prompt_processing_total_average_ms: f64,
    prompt_processing_average_samples: usize,
}

impl AppState {
    fn new() -> Self {
        Self {
            generated_text: String::new(),
            iteration_latency: RunningAverage::default(),
            prompt_token_count: 0,
            should_quit: false,
            status: "Initializing...".to_string(),
            memory_rows: Vec::new(),
            prompt_processing_time: Duration::default(),
            latency_tree: Vec::new(),
            metrics_view: MetricsView::Memory,
            latency_collapse_depth: CollapseDepth::new(),
            memory_collapse_depth: MemoryCollapseDepth::new(),
            focus: FocusArea::GeneratedText,
            text_scroll: 0,
            metrics_scroll: 0,
            text_follow_bottom: true,
            request_follow_text: false,
            text_area: Rect::new(0, 0, 0, 0),
            metrics_area: Rect::new(0, 0, 0, 0),
            alert_queue: VecDeque::new(),
            active_alert: None,
            prompt_processing_total_last_ms: 0.0,
            prompt_processing_total_average_ms: 0.0,
            prompt_processing_average_samples: 0,
        }
    }

    fn add_latency_metric(&mut self, row: LatencyRow) {
        let segments: Vec<&str> = row.label.split("::").collect();
        if segments.is_empty() {
            return;
        }

        // Check if this is a prompt processing related metric
        // Look for "Prompt Processing" anywhere in the path
        if segments.contains(&PROMPT_PROCESSING_LABEL) {
            // For prompt processing operations, just accumulate the time
            // but don't add individual sub-operations to maintain a single line display
            self.prompt_processing_total_last_ms += row.last_ms;

            // Update the running average: (old_total + new_value) / (count + 1)
            let new_average_total =
                self.prompt_processing_total_average_ms * (self.prompt_processing_average_samples as f64) + row.average_ms;
            self.prompt_processing_average_samples += 1;
            self.prompt_processing_total_average_ms = new_average_total / (self.prompt_processing_average_samples as f64);

            // Update the Prompt Processing entry with the accumulated time
            self.update_prompt_processing_entry();

            // Don't add sub-items for prompt processing - this keeps it as a single line
        } else {
            // For non-prompt processing metrics, add normally
            self.upsert_latency_path(&segments, row.last_ms, row.average_ms);
        }

        // Recalculate max depth for latency metrics since the tree structure may have changed
        self.latency_collapse_depth.calculate_max_depth(&self.latency_tree);
    }

    fn update_prompt_processing_entry(&mut self) {
        // Find if there's already a root-level "Prompt Processing" entry and update it
        // This is a simplified approach that creates "Prompt Processing" at the root level
        // with the aggregated time
        match self.latency_tree.iter_mut().find(|metric| metric.label == PROMPT_PROCESSING_LABEL) {
            Some(metric) => {
                // Update existing entry with accumulated time
                metric.last_ms = self.prompt_processing_total_last_ms;
                metric.average_ms = self.prompt_processing_total_average_ms;
            }
            None => {
                // Create new "Prompt Processing" entry at root level with accumulated time
                self.latency_tree.push(HierarchicalMetric::new(
                    PROMPT_PROCESSING_LABEL.to_string(),
                    self.prompt_processing_total_last_ms,
                    self.prompt_processing_total_average_ms,
                ));
            }
        }
    }

    fn upsert_latency_path(&mut self, path: &[&str], last_ms: f64, average_ms: f64) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let entry = self.latency_tree.iter_mut().position(|metric| metric.label == label);

        let metric = if let Some(idx) = entry {
            &mut self.latency_tree[idx]
        } else {
            self.latency_tree.push(HierarchicalMetric::new(label.to_string(), 0.0, 0.0));
            self.latency_tree
                .last_mut()
                .expect("latency_tree cannot be empty immediately after push")
        };

        if path.len() == 1 {
            metric.last_ms = last_ms;
            metric.average_ms = average_ms;
            return;
        }

        metric.upsert_path(&path[1..], last_ms, average_ms);
    }

    fn update_generation_metrics(&mut self, iteration: Option<Duration>) {
        let Some(iteration) = iteration else {
            return;
        };

        if iteration.is_zero() {
            return;
        }

        self.iteration_latency.record(iteration.as_secs_f64() * 1000.0);
    }

    fn reset_generation_metrics(&mut self) {
        self.iteration_latency.reset();
    }

    fn reset_prompt_processing_metrics(&mut self) {
        self.prompt_processing_total_last_ms = 0.0;
        self.prompt_processing_total_average_ms = 0.0;
        self.prompt_processing_average_samples = 0;

        // Remove any existing Prompt Processing entry from the tree
        self.latency_tree.retain(|metric| metric.label != PROMPT_PROCESSING_LABEL);
    }

    fn throughput_display(&self) -> String {
        if self.iteration_latency.has_samples() {
            let average_ms = self.iteration_latency.average();
            if average_ms > 0.0 {
                let tokens_per_second = 1000.0 / average_ms;
                format!("{:.1} tok/s ({:.1}ms)", tokens_per_second, average_ms)
            } else {
                "-- tok/s (--ms)".to_string()
            }
        } else {
            "-- tok/s (--ms)".to_string()
        }
    }

    fn push_alert(&mut self, alert: Alert) {
        if self.active_alert.is_some() {
            self.alert_queue.push_back(alert);
        } else {
            self.active_alert = Some(alert);
        }
    }

    fn active_alert(&self) -> Option<&Alert> {
        self.active_alert.as_ref()
    }

    fn has_active_alert(&self) -> bool {
        self.active_alert.is_some()
    }

    fn dismiss_active_alert(&mut self) {
        if let Some(next) = self.alert_queue.pop_front() {
            self.active_alert = Some(next);
        } else {
            self.active_alert = None;
        }
    }

    fn pending_alert_count(&self) -> usize {
        self.alert_queue.len()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MetricsView {
    Memory,
    Latency,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CollapseDepth {
    current: usize,
    max: Option<usize>,
}

impl CollapseDepth {
    fn new() -> Self {
        Self {
            current: usize::MAX, // Start fully uncollapsed (show all levels)
            max: None,           // Will be calculated dynamically based on data
        }
    }

    fn calculate_max_depth(&mut self, metrics: &[HierarchicalMetric]) {
        let max_found = find_max_depth(metrics, 0);
        self.max = Some(max_found);
        // If current is set to MAX (default), set it to the max found
        if self.current == usize::MAX {
            self.current = max_found;
        } else {
            // Ensure current doesn't exceed the max
            self.current = self.current.min(max_found);
        }
    }

    fn next(&mut self) {
        if let Some(max) = self.max {
            if max == 0 {
                // If max depth is 0, just toggle between 0 and 0
                self.current = 0;
            } else if self.current == 0 {
                // If at min depth (0), go back to max-1
                self.current = max - 1;
            } else if self.current == max {
                // If at max, go to max-1
                self.current = max - 1;
            } else {
                // Otherwise, decrease depth by 1
                self.current -= 1;
            }
        } else {
            // If max hasn't been calculated yet, go to 0
            self.current = 0;
        }
    }

    fn get_current_depth(&self) -> usize {
        self.current
    }

    fn label(&self) -> String {
        match self.max {
            Some(max) => {
                if self.current > max {
                    format!("Level {}", max)
                } else {
                    format!("Level {}", self.current)
                }
            }
            None => "Unknown".to_string(),
        }
    }
}

fn find_max_depth(metrics: &[HierarchicalMetric], current_depth: usize) -> usize {
    let mut max_depth = current_depth;
    for metric in metrics {
        let child_depth = find_max_depth(&metric.children, current_depth + 1);
        max_depth = max_depth.max(child_depth);
    }
    max_depth
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MemoryCollapseDepth {
    current: u8,
    max: Option<u8>,
}

impl MemoryCollapseDepth {
    fn new() -> Self {
        Self {
            current: u8::MAX, // Start fully uncollapsed (show all levels)
            max: None,        // Will be calculated dynamically based on data
        }
    }

    fn calculate_max_depth(&mut self, rows: &[MemoryRow]) {
        let max_found = rows.iter().map(|row| row.level).max().unwrap_or(0);
        self.max = Some(max_found);
        // If current is set to MAX (default), set it to the max found
        if self.current == u8::MAX {
            self.current = max_found;
        } else {
            // Ensure current doesn't exceed the max
            self.current = self.current.min(max_found);
        }
    }

    fn next(&mut self) {
        if let Some(max) = self.max {
            if max == 0 {
                // If max depth is 0, just toggle between 0 and 0
                self.current = 0;
            } else if self.current == 0 {
                // If at min depth (0), go back to max-1
                self.current = max - 1;
            } else if self.current == max {
                // If at max, go to max-1
                self.current = max - 1;
            } else {
                // Otherwise, decrease depth by 1
                self.current -= 1;
            }
        } else {
            // If max hasn't been calculated yet, go to 0
            self.current = 0;
        }
    }

    fn get_current_depth(&self) -> u8 {
        self.current
    }

    fn label(&self) -> String {
        match self.max {
            Some(max) => {
                if self.current > max {
                    format!("Level {}", max)
                } else {
                    format!("Level {}", self.current)
                }
            }
            None => "Unknown".to_string(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FocusArea {
    GeneratedText,
    Metrics,
}

impl AppState {
    fn focus_next(&mut self) {
        self.focus = match self.focus {
            FocusArea::GeneratedText => FocusArea::Metrics,
            FocusArea::Metrics => FocusArea::GeneratedText,
        };
    }

    fn focus_prev(&mut self) {
        self.focus_next();
    }

    fn scroll_active(&mut self, delta: i32) {
        match self.focus {
            FocusArea::GeneratedText => {
                if delta != 0 {
                    self.text_follow_bottom = false;
                }
                Self::adjust_scroll(&mut self.text_scroll, delta);
                if self.text_scroll == 0 && delta <= 0 {
                    self.text_follow_bottom = true;
                }
            }
            FocusArea::Metrics => {
                Self::adjust_scroll(&mut self.metrics_scroll, delta);
            }
        }
    }

    fn scroll_active_to_start(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.text_scroll = 0;
                self.text_follow_bottom = false;
            }
            FocusArea::Metrics => self.metrics_scroll = 0,
        }
    }

    fn scroll_active_to_end(&mut self) {
        match self.focus {
            FocusArea::GeneratedText => {
                self.text_follow_bottom = true;
                self.request_follow_text = true;
            }
            FocusArea::Metrics => {
                self.metrics_scroll = u16::MAX;
            }
        }
    }

    fn adjust_scroll(value: &mut u16, delta: i32) {
        let current = i32::from(*value);
        let next = (current + delta).clamp(0, u16::MAX as i32);
        *value = next as u16;
    }

    fn toggle_collapse(&mut self) {
        match self.metrics_view {
            MetricsView::Latency => self.latency_collapse_depth.next(),
            MetricsView::Memory => self.memory_collapse_depth.next(),
        }
        self.reset_metrics_scroll();
    }

    fn reset_metrics_scroll(&mut self) {
        self.metrics_scroll = 0;
    }

    fn handle_click(&mut self, column: u16, row: u16) {
        if rect_contains(self.text_area, column, row) {
            self.focus = FocusArea::GeneratedText;
        } else if rect_contains(self.metrics_area, column, row) {
            self.focus = FocusArea::Metrics;
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

fn ui(frame: &mut Frame, state: &mut AppState) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());

    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(main_layout[0]);

    let text_block = Block::default()
        .title("Generated Text (q to quit)")
        .borders(Borders::ALL)
        .border_style(border_style(state.focus == FocusArea::GeneratedText));
    let text_area_widget = Paragraph::new(state.generated_text.clone())
        .block(text_block)
        .wrap(Wrap { trim: false })
        .scroll((state.text_scroll, 0));

    let sidebar_block = Block::default().title("Metrics").borders(Borders::ALL);
    let sidebar_area = body_layout[1];
    let sidebar_inner = sidebar_block.inner(sidebar_area);

    let sidebar_sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(0)])
        .split(sidebar_inner);

    let prompt_section = Paragraph::new(format!(
        "Prompt Tokens: {}\nProcessing Time: {}",
        state.prompt_token_count,
        format_duration(state.prompt_processing_time)
    ))
    .block(Block::default().title("Prompt").borders(Borders::ALL));

    let metrics_block_title = match state.metrics_view {
        MetricsView::Memory => "Memory Usage",
        MetricsView::Latency => "Latency",
    };

    let collapse_label = match state.metrics_view {
        MetricsView::Memory => state.memory_collapse_depth.label(),
        MetricsView::Latency => state.latency_collapse_depth.label(),
    };
    let metrics_help = format!("[m] Memory [l] Latency [c] Collapse ({})", collapse_label);

    // Always calculate and update max depths before rendering to account for dynamic changes
    if matches!(state.metrics_view, MetricsView::Memory) && !state.memory_rows.is_empty() {
        state.memory_collapse_depth.calculate_max_depth(&state.memory_rows);
    } else if matches!(state.metrics_view, MetricsView::Latency) && !state.latency_tree.is_empty() {
        state.latency_collapse_depth.calculate_max_depth(&state.latency_tree);
    }

    let metrics_text = match state.metrics_view {
        MetricsView::Memory => render_memory_metrics(&state.memory_rows, state.memory_collapse_depth.get_current_depth()),
        MetricsView::Latency => render_hierarchical_latency_metrics(&state.latency_tree, state.latency_collapse_depth.get_current_depth()),
    };

    let metrics_block = Block::default()
        .title(metrics_block_title)
        .borders(Borders::ALL)
        .border_style(border_style(state.focus == FocusArea::Metrics));
    let metrics_section = Paragraph::new(format!("{}\n\n{}", metrics_help, metrics_text))
        .block(metrics_block)
        .wrap(Wrap { trim: false })
        .scroll((state.metrics_scroll, 0));

    let status_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(main_layout[1]);

    let status_text = Paragraph::new(state.status.clone()).style(Style::default().fg(Color::White).bg(Color::Blue));

    let throughput_text = Paragraph::new(state.throughput_display())
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Right);

    frame.render_widget(text_area_widget, body_layout[0]);
    frame.render_widget(sidebar_block, sidebar_area);
    frame.render_widget(prompt_section, sidebar_sections[0]);
    frame.render_widget(metrics_section, sidebar_sections[1]);
    frame.render_widget(status_text, status_layout[0]);
    frame.render_widget(throughput_text, status_layout[1]);

    state.text_area = body_layout[0];
    state.metrics_area = sidebar_sections[1];

    if let Some(alert) = state.active_alert() {
        render_alert_modal(frame, alert, state.pending_alert_count());
    }

    if state.request_follow_text {
        if state.text_area.height > 0 {
            let content_lines = state.generated_text.matches('\n').count() + 1;
            let visible = state.text_area.height as usize;
            let baseline = content_lines.saturating_sub(visible) as u16;
            state.text_scroll = baseline;
        }
        state.request_follow_text = false;
    }
}

fn render_memory_metrics(rows: &[MemoryRow], collapse_depth: u8) -> String {
    if rows.is_empty() {
        return "Collecting data...".to_string();
    }

    rows.iter()
        .filter(|row| row.level <= collapse_depth)
        .map(|row| {
            let indent = "  ".repeat(row.level as usize);
            let mut line = format!(
                "{}{}: {}",
                indent,
                row.label,
                format_current_and_peak(row.current_total_mb, row.peak_total_mb)
            );

            let mut deltas = Vec::new();
            if row.current_pool_mb > 0.0 || row.peak_pool_mb > 0.0 {
                deltas.push(format!("pool {}", format_current_and_peak(row.current_pool_mb, row.peak_pool_mb)));
            }
            if row.current_kv_mb > 0.0 || row.peak_kv_mb > 0.0 {
                deltas.push(format!("kv {}", format_current_and_peak(row.current_kv_mb, row.peak_kv_mb)));
            }
            if row.current_kv_cache_mb > 0.0 || row.peak_kv_cache_mb > 0.0 {
                deltas.push(format!(
                    "kv-cache {}",
                    format_current_and_peak(row.current_kv_cache_mb, row.peak_kv_cache_mb)
                ));
            }
            if !deltas.is_empty() {
                line.push_str(&format!(" | {}", deltas.join(", ")));
            }

            if row.show_absolute {
                let mut absolutes = Vec::new();
                if row.absolute_pool_mb > 0.0 {
                    absolutes.push(format!("pool {}", format_memory_amount(row.absolute_pool_mb)));
                }
                if row.absolute_kv_mb > 0.0 {
                    absolutes.push(format!("kv {}", format_memory_amount(row.absolute_kv_mb)));
                }
                if row.absolute_kv_cache_mb > 0.0 {
                    absolutes.push(format!("kv-cache {}", format_memory_amount(row.absolute_kv_cache_mb)));
                }
                if !absolutes.is_empty() {
                    line.push_str(&format!(" | abs {}", absolutes.join(", ")));
                }
            }

            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_hierarchical_latency_metrics(metrics: &[HierarchicalMetric], max_depth: usize) -> String {
    if metrics.is_empty() {
        return "Collecting data...".to_string();
    }

    let mut lines = Vec::new();
    for metric in metrics {
        render_metric(metric, 0, max_depth, &mut lines);
    }

    lines.join("\n")
}

fn render_metric(metric: &HierarchicalMetric, depth: usize, max_depth: usize, lines: &mut Vec<String>) {
    if depth > max_depth {
        return;
    }

    let level = u8::try_from(depth).unwrap_or(u8::MAX);
    // Use inclusive timing for display (parent + all descendants) to show total impact
    let (inclusive_last_ms, inclusive_average_ms) = metric.get_inclusive_timing();
    lines.push(format_latency_line(level, &metric.label, inclusive_last_ms, inclusive_average_ms));

    if depth == max_depth {
        return;
    }

    for child in &metric.children {
        render_metric(child, depth + 1, max_depth, lines);
    }
}

fn format_latency_line(level: u8, label: &str, last_ms: f64, avg_ms: f64) -> String {
    let indent = "  ".repeat(level as usize);
    format!("{}{} - {:.2}ms ({:.2} avg)", indent, label, last_ms, avg_ms)
}

fn format_current_and_peak(current_mb: f64, peak_mb: f64) -> String {
    let current = format_memory_amount(current_mb);
    if nearly_equal(current_mb, peak_mb) {
        current
    } else {
        format!("{} (peak {})", current, format_memory_amount(peak_mb))
    }
}

fn format_memory_amount(mb: f64) -> String {
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else if mb >= 1.0 {
        format!("{:.2} MB", mb)
    } else if mb > 0.0 {
        let kb = mb * 1024.0;
        if kb >= 10.0 {
            format!("{:.0} KB", kb)
        } else {
            format!("{:.2} KB", kb)
        }
    } else {
        "0 MB".to_string()
    }
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.005
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.0}ms", duration.as_secs_f64() * 1000.0)
    } else if duration.as_nanos() > 0 {
        "<1ms".to_string()
    } else {
        "0ms".to_string()
    }
}

fn border_style(active: bool) -> Style {
    if active {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}

fn rect_contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x && x < rect.x.saturating_add(rect.width) && y >= rect.y && y < rect.y.saturating_add(rect.height)
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

fn render_alert_modal(frame: &mut Frame, alert: &Alert, pending: usize) {
    let area = centered_rect(60, 40, frame.area());
    frame.render_widget(Clear, area);

    let timestamp = alert.timestamp.to_rfc3339_opts(SecondsFormat::Secs, true);
    let mut message = format!("{}\n\nTime: {}", alert.message, timestamp);
    if pending > 0 {
        message.push_str(&format!("\n\n{} more alert(s) pending", pending));
    }
    message.push_str("\n\nPress Enter, Space, or Esc to dismiss.");

    let block = Block::default()
        .title(alert.level.as_str().to_string())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(alert_color(&alert.level)));

    let paragraph = Paragraph::new(message)
        .block(block)
        .wrap(Wrap { trim: false })
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White).bg(Color::Black));

    frame.render_widget(paragraph, area);
}

fn alert_color(level: &AlertLevel) -> Color {
    match level {
        AlertLevel::Info => Color::LightBlue,
        AlertLevel::Warning => Color::Yellow,
        AlertLevel::Error => Color::Red,
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let remaining_y = 100u16.saturating_sub(percent_y);
    let top_margin = remaining_y / 2;
    let bottom_margin = remaining_y.saturating_sub(top_margin);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(top_margin),
            Constraint::Percentage(percent_y.min(100)),
            Constraint::Percentage(bottom_margin),
        ])
        .split(area);

    let remaining_x = 100u16.saturating_sub(percent_x);
    let left_margin = remaining_x / 2;
    let right_margin = remaining_x.saturating_sub(left_margin);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(left_margin),
            Constraint::Percentage(percent_x.min(100)),
            Constraint::Percentage(right_margin),
        ])
        .split(vertical[1])[1]
}
