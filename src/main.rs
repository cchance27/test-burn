use anyhow::Result;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{env, io::stdout, process, sync::mpsc, thread, time::Duration};

use test_burn::{
    app_event::{AppEvent, LatencyRow, MemoryRow},
    gguf::{model_loader::GGUFModelLoader, GGUFFile},
    metallic::{
        generation::{generate_streaming, GenerationConfig},  Context, ContextConfig, Dtype, F32Element, Tokenizer
    },
};

fn main() -> Result<()> {
    // Minimal CLI:
    //   cargo run -- /path/to/model.gguf [PROMPT]
    let mut args = env::args().skip(1);
    let gguf_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run -- <GGUF_PATH> [PROMPT]");
            process::exit(1);
        }
    };
    let prompt = args.next().unwrap_or_else(|| "Create a short javascript hello world app.".to_string());

    let (tx, rx) = mpsc::channel();

    let generation_handle = thread::spawn(move || -> Result<()> {
        tx.send(AppEvent::StatusUpdate("Loading GGUF Metadata...".to_string()))?;
        let gguf = GGUFFile::load_mmap_and_get_metadata(&gguf_path)?;

        tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
        let mut ctx= Context::<F32Element>::with_config(ContextConfig::new(Dtype::F32))?;

        tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
        let loader = GGUFModelLoader::new(gguf);
        let mapped_bytes = loader.mapped_len();
        let host_overheads = if mapped_bytes > 0 {
            vec![("GGUF File MMAP".to_string(), mapped_bytes)]
        } else {
            Vec::new()
        };
        let gguf_model = loader.load_model(&ctx)?;

        tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
        let mut qwen = gguf_model.instantiate(&mut ctx)?;

        tx.send(AppEvent::StatusUpdate("Initializing tokenizer...".to_string()))?;
        let tokenizer = Tokenizer::from_gguf_metadata(&gguf_model.metadata)?;

        tx.send(AppEvent::StatusUpdate("Encoding prompt...".to_string()))?;
        let tokens = tokenizer.encode(&prompt)?;
        tx.send(AppEvent::TokenCount(tokens.len()))?;

        let cfg = GenerationConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
        };

        tx.send(AppEvent::StatusUpdate("Generating...".to_string()))?;
        let _ = generate_streaming(&mut qwen, &tokenizer, &mut ctx, &prompt, &cfg, &tx, &host_overheads);
        tx.send(AppEvent::StatusUpdate("Done.".to_string()))?;
        Ok(())
    });

    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::new();

    while !app_state.should_quit {
        if crossterm::event::poll(Duration::from_millis(50))? {
            match crossterm::event::read()? {
                crossterm::event::Event::Key(key) => match key.code {
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
                },
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
    metrics_collapsed: bool,
    focus: FocusArea,
    text_scroll: u16,
    metrics_scroll: u16,
    text_follow_bottom: bool,
    request_follow_text: bool,
    text_area: Rect,
    metrics_area: Rect,
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
            metrics_collapsed: false,
            focus: FocusArea::GeneratedText,
            text_scroll: 0,
            metrics_scroll: 0,
            text_follow_bottom: true,
            request_follow_text: false,
            text_area: Rect::new(0, 0, 0, 0),
            metrics_area: Rect::new(0, 0, 0, 0),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MetricsView {
    Memory,
    Latency,
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
        self.metrics_collapsed = !self.metrics_collapsed;
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

    let metrics_help = "[m] Memory [l] Latency [c] Collapse";

    let metrics_text = match state.metrics_view {
        MetricsView::Memory => {
            if state.memory_rows.is_empty() {
                "Collecting data...".to_string()
            } else {
                let collapse_depth = 1;
                state
                    .memory_rows
                    .iter()
                    .filter(|row| !state.metrics_collapsed || row.level <= collapse_depth)
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
        }
        MetricsView::Latency => {
            if state.latency_rows.is_empty() {
                "Collecting data...".to_string()
            } else {
                let collapse_depth = 1;
                state
                    .latency_rows
                    .iter()
                    .filter(|row| !state.metrics_collapsed || row.level <= collapse_depth)
                    .map(|row| {
                        let indent = "  ".repeat(row.level as usize);
                        format!("{}{} - {:.2}ms ({:.2} avg)", indent, row.label, row.last_ms, row.average_ms)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
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
