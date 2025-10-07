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
    level: u8,
    children: Vec<HierarchicalMetric>,
}

impl HierarchicalMetric {
    fn new(label: String, last_ms: f64, average_ms: f64, level: u8) -> Self {
        Self {
            label,
            last_ms,
            average_ms,
            level,
            children: Vec::new(),
        }
    }

    fn ensure_child(&mut self, label: &str, level: u8) -> &mut HierarchicalMetric {
        if let Some(position) = self.children.iter().position(|child| child.label == label) {
            &mut self.children[position]
        } else {
            self.children.push(HierarchicalMetric::new(label.to_string(), 0.0, 0.0, level));
            self.children
                .last_mut()
                .expect("children vector cannot be empty immediately after push")
        }
    }

    fn upsert_path(&mut self, path: &[&str], last_ms: f64, average_ms: f64, level: u8) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let child = self.ensure_child(label, level);

        if path.len() == 1 {
            child.last_ms = last_ms;
            child.average_ms = average_ms;
        } else {
            child.upsert_path(&path[1..], last_ms, average_ms, level + 1);
        }
    }
}

const GENERATION_LOOP_LABEL: &str = "Generation Loop";
const PROMPT_PROCESSING_LABEL: &str = "Prompt Processing";

const BLOCK_STAGE_PREFIXES: &[(&str, &str)] = &[
    ("attn_norm_block_", "attn_norm"),
    ("qkv_proj_block_", "attn_qkv_proj"),
    ("attn_rearrange_block_", "attn_rearrange"),
    ("rope_block_", "Rope"),
    ("kv_cache_block_", "kv_cache"),
    ("kv_repeat_block_", "kv_repeat"),
    ("sdpa_block_", "Sdpa"),
    ("attn_output_block_", "attn_output"),
    ("mlp_norm_block_", "mlp_norm"),
    ("mlp_swiglu_block_", "mlp_swiglu"),
    ("mlp_output_block_", "mlp_output"),
];

fn metric_event_to_latency_rows(event: &MetricEvent) -> Vec<LatencyRow> {
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
            "embedding" => Some(vec![GENERATION_LOOP_LABEL.to_string(), "Embedding".to_string()]),
            _ => None,
        },
        _ => None,
    }
}

fn map_gpu_op_completed(op_name: &str) -> Option<Vec<String>> {
    let base_name = op_name.split('#').next().unwrap_or(op_name);
    if let Some(segments) = map_generation_stage(base_name) {
        return Some(segments);
    }
    if let Some(segments) = map_block_stage(base_name) {
        return Some(segments);
    }
    map_prompt_stage(base_name)
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

fn map_prompt_stage(name: &str) -> Option<Vec<String>> {
    if let Some(rest) = name.strip_prefix("prompt_step_")
        && rest.parse::<usize>().is_ok()
    {
        return Some(vec![PROMPT_PROCESSING_LABEL.to_string()]);
    }
    None
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
        //console_exporter
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
                    app_state.prompt_token_count = count;
                }
                AppEvent::StatusUpdate(status) => {
                    app_state.status = status;
                }
                AppEvent::MemoryUpdate(memory_rows) => {
                    app_state.memory_rows = memory_rows;
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
    metrics_collapse_state: CollapseState,
    focus: FocusArea,
    text_scroll: u16,
    metrics_scroll: u16,
    text_follow_bottom: bool,
    request_follow_text: bool,
    text_area: Rect,
    metrics_area: Rect,
    alert_queue: VecDeque<Alert>,
    active_alert: Option<Alert>,
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
            metrics_collapse_state: CollapseState::Collapsed,
            focus: FocusArea::GeneratedText,
            text_scroll: 0,
            metrics_scroll: 0,
            text_follow_bottom: true,
            request_follow_text: false,
            text_area: Rect::new(0, 0, 0, 0),
            metrics_area: Rect::new(0, 0, 0, 0),
            alert_queue: VecDeque::new(),
            active_alert: None,
        }
    }

    fn add_latency_metric(&mut self, row: LatencyRow) {
        let segments: Vec<&str> = row.label.split("::").collect();
        if segments.is_empty() {
            return;
        }

        self.upsert_latency_path(&segments, row.last_ms, row.average_ms, 0);
    }

    fn upsert_latency_path(&mut self, path: &[&str], last_ms: f64, average_ms: f64, level: u8) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let entry = self.latency_tree.iter_mut().position(|metric| metric.label == label);

        let metric = if let Some(idx) = entry {
            &mut self.latency_tree[idx]
        } else {
            self.latency_tree.push(HierarchicalMetric::new(label.to_string(), 0.0, 0.0, level));
            self.latency_tree
                .last_mut()
                .expect("latency_tree cannot be empty immediately after push")
        };

        if path.len() == 1 {
            metric.last_ms = last_ms;
            metric.average_ms = average_ms;
            return;
        }

        metric.upsert_path(&path[1..], last_ms, average_ms, level + 1);
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
enum CollapseState {
    Uncollapsed,
    Collapsed,
    VeryCollapsed,
}

impl CollapseState {
    fn next(self) -> Self {
        match self {
            Self::Uncollapsed => Self::Collapsed,
            Self::Collapsed => Self::VeryCollapsed,
            Self::VeryCollapsed => Self::Uncollapsed,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Uncollapsed => "Uncollapsed",
            Self::Collapsed => "Collapsed",
            Self::VeryCollapsed => "Very Collapsed",
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
        self.metrics_collapse_state = self.metrics_collapse_state.next();
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

    let metrics_help = format!("[m] Memory [l] Latency [c] Collapse ({})", state.metrics_collapse_state.label());

    let metrics_text = match state.metrics_view {
        MetricsView::Memory => render_memory_metrics(&state.memory_rows, state.metrics_collapse_state),
        MetricsView::Latency => render_hierarchical_latency_metrics(&state.latency_tree, state.metrics_collapse_state),
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

fn render_memory_metrics(rows: &[MemoryRow], collapse_state: CollapseState) -> String {
    if rows.is_empty() {
        return "Collecting data...".to_string();
    }

    let collapse_depth = match collapse_state {
        CollapseState::Uncollapsed => u8::MAX,
        CollapseState::Collapsed => 1,
        CollapseState::VeryCollapsed => 0,
    };

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

fn render_hierarchical_latency_metrics(metrics: &[HierarchicalMetric], collapse_state: CollapseState) -> String {
    if metrics.is_empty() {
        return "Collecting data...".to_string();
    }

    let max_depth = match collapse_state {
        CollapseState::Uncollapsed => usize::MAX,
        CollapseState::Collapsed => 2,
        CollapseState::VeryCollapsed => 0,
    };

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
    lines.push(format_latency_line(level, &metric.label, metric.last_ms, metric.average_ms));

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
