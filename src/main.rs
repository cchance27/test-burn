use anyhow::Result;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{env, io::stdout, process, sync::mpsc, thread, time::Duration};

use test_burn::{
    app_event::{AppEvent, LatencyRow},
    gguf,
    metallic::{
        generation::{generate_streaming, GenerationConfig},
        models::Qwen25,
        Context, Tokenizer,
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
    let prompt = args.next().unwrap_or_else(|| "Hello World".to_string());

    let (tx, rx) = mpsc::channel();

    let generation_handle = thread::spawn(move || -> Result<()> {
        tx.send(AppEvent::StatusUpdate("Loading GGUF...".to_string()))?;
        let gguf = gguf::GGUFFile::load(&gguf_path)?;

        tx.send(AppEvent::StatusUpdate("Initializing context...".to_string()))?;
        let mut ctx = Context::new()?;

        tx.send(AppEvent::StatusUpdate("Loading model...".to_string()))?;
        let loader = gguf::model_loader::GGUFModelLoader::new(gguf);
        let gguf_model = loader.load_model(&ctx)?;

        tx.send(AppEvent::StatusUpdate("Instantiating model...".to_string()))?;
        let mut qwen: Qwen25 = gguf_model.instantiate(&mut ctx)?;

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
        let _ = generate_streaming(&mut qwen, &tokenizer, &mut ctx, &prompt, &cfg, &tx);
        tx.send(AppEvent::StatusUpdate("Done.".to_string()))?;
        Ok(())
    });

    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::new();

    while !app_state.should_quit {
        if crossterm::event::poll(Duration::from_millis(50))?
            && let crossterm::event::Event::Key(key) = crossterm::event::read()?
            && key.code == crossterm::event::KeyCode::Char('q')
        {
            app_state.should_quit = true;
        }

        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token {
                    text,
                    tokens_per_second,
                    prompt_processing,
                    generation,
                } => {
                    app_state.generated_text.push_str(&text);
                    app_state.tokens_per_second = tokens_per_second;
                    app_state.prompt_processing_time = prompt_processing;
                    app_state.generation_time = generation;
                }
                AppEvent::TokenCount(count) => {
                    app_state.prompt_token_count = count;
                }
                AppEvent::StatusUpdate(status) => {
                    app_state.status = status;
                }
                AppEvent::MemoryUpdate(memory_usage) => {
                    app_state.memory_usage = memory_usage;
                }
                AppEvent::LatencyUpdate(rows) => {
                    app_state.latency_rows = rows;
                }
            }
        }

        terminal.draw(|frame| ui(frame, &app_state))?;
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
    memory_usage: String,
    prompt_processing_time: Duration,
    generation_time: Duration,
    latency_rows: Vec<LatencyRow>,
}

impl AppState {
    fn new() -> Self {
        Self {
            generated_text: String::new(),
            tokens_per_second: 0.0,
            prompt_token_count: 0,
            should_quit: false,
            status: "Initializing...".to_string(),
            memory_usage: String::new(),
            prompt_processing_time: Duration::default(),
            generation_time: Duration::default(),
            latency_rows: Vec::new(),
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

fn ui(frame: &mut Frame, state: &AppState) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());

    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(main_layout[0]);

    let text_area = Paragraph::new(state.generated_text.clone())
        .block(Block::default().title("Generated Text (q to quit)").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    let sidebar_block = Block::default().title("Metrics").borders(Borders::ALL);
    let sidebar_area = body_layout[1];
    let sidebar_inner = sidebar_block.inner(sidebar_area);

    let sidebar_sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Min(5),
        ])
        .split(sidebar_inner);

    let memory_text = if state.memory_usage.is_empty() {
        "Collecting data...".to_string()
    } else {
        state.memory_usage.clone()
    };
    let memory_section = Paragraph::new(memory_text).block(Block::default().title("Memory Usage").borders(Borders::ALL));

    let latency_text = if state.latency_rows.is_empty() {
        "Collecting data...".to_string()
    } else {
        state
            .latency_rows
            .iter()
            .map(|row| {
                let indent = "  ".repeat(row.level as usize);
                format!("{}{} - {:.2}ms ({:.2} avg)", indent, row.label, row.last_ms, row.average_ms)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let latency_section = Paragraph::new(latency_text).block(Block::default().title("Latency").borders(Borders::ALL));

    let prompt_section = Paragraph::new(format!(
        "Prompt Tokens: {}\nProcessing Time: {}",
        state.prompt_token_count,
        format_duration(state.prompt_processing_time)
    ))
    .block(Block::default().title("Prompt").borders(Borders::ALL));

    let generation_section = Paragraph::new(format!(
        "Total Time: {}\nThroughput: {:.2} it/s",
        format_duration(state.generation_time),
        state.tokens_per_second
    ))
    .block(Block::default().title("Generation").borders(Borders::ALL));

    let status_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(main_layout[1]);

    let status_text = Paragraph::new(state.status.clone()).style(Style::default().fg(Color::White).bg(Color::Blue));

    let throughput_text = Paragraph::new(format!("{:.2} it/s", state.tokens_per_second))
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Right);

    frame.render_widget(text_area, body_layout[0]);
    frame.render_widget(sidebar_block, sidebar_area);
    frame.render_widget(memory_section, sidebar_sections[0]);
    frame.render_widget(latency_section, sidebar_sections[1]);
    frame.render_widget(prompt_section, sidebar_sections[2]);
    frame.render_widget(generation_section, sidebar_sections[3]);
    frame.render_widget(status_text, status_layout[0]);
    frame.render_widget(throughput_text, status_layout[1]);
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
