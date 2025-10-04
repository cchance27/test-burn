use anyhow::Result;
use chrono::SecondsFormat;
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

use test_burn::{
    alert,
    app_event::{Alert, AlertLevel, AppEvent, LatencyRow, MemoryRow},
    gguf::{GGUFFile, model_loader::GGUFModelLoader},
    metallic::{
        Context, F16Element, Tokenizer,
        generation::{GenerationConfig, generate_streaming},
        metrics::{
            MemoryBlockStat, MemoryScopeStat, ModelMemoryNode, ProcessMemoryTracker, ScalarStat, build_memory_rows,
            build_model_memory_tree, sample_process_memory,
        },
    },
};

fn main() -> Result<()> {
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
                let mut process_memory_tracker = ProcessMemoryTracker::new();
                let mut host_memory = ScalarStat::default();
                let mut model_memory_tree = ModelMemoryNode::branch("Model Weights", Vec::new());
                let mut host_overheads: Vec<(String, usize)> = Vec::new();

                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Loading GGUF Metadata...".to_string()))?;
                let gguf = GGUFFile::load_mmap_and_get_metadata(&gguf_path)?;
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
                let mut ctx = Context::<F16Element>::new()?;
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
                let loader = GGUFModelLoader::new(gguf);
                host_overheads.clear();
                let mapped_bytes = loader.mapped_len();
                if mapped_bytes > 0 {
                    host_overheads.push(("GGUF File MMAP".to_string(), mapped_bytes));
                }
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;
                let gguf_model = loader.load_model()?;
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
                let mut qwen = gguf_model.instantiate(&mut ctx)?;
                model_memory_tree = build_model_memory_tree(&qwen);
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Initializing tokenizer...".to_string()))?;
                let tokenizer = Tokenizer::from_gguf_metadata(&gguf_model.metadata)?;
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                worker_tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
                let tokens = tokenizer.encode(&prompt)?;
                worker_tx.send(AppEvent::TokenCount(tokens.len()))?;
                emit_startup_memory_update(
                    &worker_tx,
                    &mut process_memory_tracker,
                    &mut host_memory,
                    &model_memory_tree,
                    &host_overheads,
                )?;

                let cfg = GenerationConfig {
                    max_tokens: 4096,
                    temperature: 0.7,
                    top_p: 0.95,
                    top_k: 40,
                };

                worker_tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
                generate_streaming(
                    &mut qwen,
                    &tokenizer,
                    &mut ctx,
                    &prompt,
                    &cfg,
                    &worker_tx,
                    &host_overheads,
                    &mut host_memory,
                    &mut process_memory_tracker,
                )?;
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

        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token {
                    text,
                    tokens_per_second,
                    prompt_processing,
                    generation: _,
                } => {
                    let was_following = app_state.text_follow_bottom;
                    app_state.generated_text.push_str(&text);
                    if was_following {
                        app_state.request_follow_text = true;
                    }
                    app_state.tokens_per_second = tokens_per_second;
                    app_state.prompt_processing_time = prompt_processing;
                }
                AppEvent::TokenCount(count) => {
                    app_state.prompt_token_count = count;
                }
                AppEvent::StatusUpdate(status) => {
                    app_state.status = status;
                }
                AppEvent::MemoryUpdate(memory_rows) => {
                    app_state.memory_rows = memory_rows;
                }
                AppEvent::LatencyUpdate(rows) => {
                    app_state.latency_rows = rows;
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

struct AppState {
    generated_text: String,
    tokens_per_second: f64,
    prompt_token_count: usize,
    should_quit: bool,
    status: String,
    memory_rows: Vec<MemoryRow>,
    prompt_processing_time: Duration,
    latency_rows: Vec<LatencyRow>,
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
            tokens_per_second: 0.0,
            prompt_token_count: 0,
            should_quit: false,
            status: "Initializing...".to_string(),
            memory_rows: Vec::new(),
            prompt_processing_time: Duration::default(),
            latency_rows: Vec::new(),
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

fn emit_startup_memory_update(
    tx: &mpsc::Sender<AppEvent>,
    tracker: &mut Option<ProcessMemoryTracker>,
    host_memory: &mut ScalarStat,
    model_memory_tree: &ModelMemoryNode,
    host_overheads: &[(String, usize)],
) -> Result<(), mpsc::SendError<AppEvent>> {
    sample_process_memory(tracker, host_memory);
    let empty_embed = MemoryScopeStat::default();
    let empty_forward = MemoryScopeStat::default();
    let empty_output = MemoryScopeStat::default();
    let empty_blocks: Vec<MemoryBlockStat> = Vec::new();
    let rows = build_memory_rows(
        model_memory_tree,
        host_memory,
        &empty_embed,
        &empty_forward,
        None,
        &empty_blocks,
        &empty_output,
        host_overheads,
    );
    tx.send(AppEvent::MemoryUpdate(rows))
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

    let metrics_help = format!("[m] Memory [l] Latency [c] Collapse ({})", state.metrics_collapse_state.label());

    let metrics_text = match state.metrics_view {
        MetricsView::Memory => render_memory_metrics(&state.memory_rows, state.metrics_collapse_state),
        MetricsView::Latency => render_latency_metrics(&state.latency_rows, state.metrics_collapse_state),
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

    let throughput_text = Paragraph::new(format!("{:.2} it/s", state.tokens_per_second))
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

fn render_latency_metrics(rows: &[LatencyRow], collapse_state: CollapseState) -> String {
    if rows.is_empty() {
        return "Collecting data...".to_string();
    }

    match collapse_state {
        CollapseState::Uncollapsed => render_latency_uncollapsed(rows),
        CollapseState::Collapsed => render_latency_collapsed(rows),
        CollapseState::VeryCollapsed => render_latency_very_collapsed(rows),
    }
}

fn render_latency_uncollapsed(rows: &[LatencyRow]) -> String {
    rows.iter()
        .map(|row| format_latency_line(row.level, &row.label, row.last_ms, row.average_ms))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_latency_very_collapsed(rows: &[LatencyRow]) -> String {
    rows.iter()
        .filter(|row| row.level <= 1)
        .map(|row| format_latency_line(row.level, &row.label, row.last_ms, row.average_ms))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_latency_collapsed(rows: &[LatencyRow]) -> String {
    let Some(block_start) = rows.iter().position(|row| row.level == 1 && row.label.starts_with("Block ")) else {
        return render_latency_uncollapsed(rows);
    };

    let mut block_end = block_start;
    #[allow(clippy::needless_range_loop)]
    for idx in block_start..rows.len() {
        let row = &rows[idx];
        if idx != block_start && row.level == 0 {
            break;
        }
        block_end = idx;
    }

    let mut block_total_last = 0.0;
    let mut block_total_avg = 0.0;
    let mut block_count = 0u32;
    let mut phase_totals: Vec<(String, f64, f64)> = Vec::new();

    for row in &rows[block_start..=block_end] {
        match row.level {
            1 => {
                block_count += 1;
                block_total_last += row.last_ms;
                block_total_avg += row.average_ms;
            }
            2 => {
                if let Some((_, total_last, total_avg)) = phase_totals.iter_mut().find(|(label, _, _)| label == &row.label) {
                    *total_last += row.last_ms;
                    *total_avg += row.average_ms;
                } else {
                    phase_totals.push((row.label.clone(), row.last_ms, row.average_ms));
                }
            }
            _ => {}
        }
    }

    if block_count == 0 {
        return render_latency_uncollapsed(rows);
    }

    let mut lines: Vec<String> = Vec::new();

    lines.extend(
        rows[..block_start]
            .iter()
            .map(|row| format_latency_line(row.level, &row.label, row.last_ms, row.average_ms)),
    );

    lines.push(format_latency_line(0, "Blocks", block_total_last, block_total_avg));

    for (label, total_last, total_avg) in phase_totals {
        lines.push(format_latency_line(1, &label, total_last, total_avg));
    }

    if block_end + 1 < rows.len() {
        lines.extend(
            rows[block_end + 1..]
                .iter()
                .map(|row| format_latency_line(row.level, &row.label, row.last_ms, row.average_ms)),
        );
    }

    lines.join("\n")
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
