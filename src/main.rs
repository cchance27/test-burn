use anyhow::Result;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{
    env,
    io::stdout,
    process,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use test_burn::{
    app_event::AppEvent,
    gguf,
    metallic::{
        Context, Tokenizer,
        generation::{GenerationConfig, generate_streaming},
        models::Qwen25,
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
        let start_time = Instant::now();

        let _ = generate_streaming(&mut qwen, &tokenizer, &mut ctx, &prompt, &cfg, &tx, start_time);
        tx.send(AppEvent::StatusUpdate("Done.".to_string()))?;
        Ok(())
    });

    let mut terminal = setup_terminal()?;
    let mut app_state = AppState::new();

    while !app_state.should_quit {
        if crossterm::event::poll(Duration::from_millis(50))?
            && let crossterm::event::Event::Key(key) = crossterm::event::read()?
                && key.code == crossterm::event::KeyCode::Char('q') {
                    app_state.should_quit = true;
                }

        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::Token(token, tokens_per_second) => {
                    app_state.generated_text.push_str(&token);
                    app_state.tokens_per_second = tokens_per_second;
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

    let text_area = Paragraph::new(state.generated_text.clone())
        .block(Block::default().title("Generated Text (q to quit)").borders(Borders::ALL))
        .wrap(Wrap { trim: false });

    let status_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(33), Constraint::Percentage(34), Constraint::Percentage(33)])
        .split(main_layout[1]);

    let status_text = Paragraph::new(state.status.clone()).style(Style::default().fg(Color::White).bg(Color::Blue));

    let memory_text = Paragraph::new(state.memory_usage.clone())
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Center);

    let its_text = Paragraph::new(format!("it/s: {:.2}", state.tokens_per_second))
        .style(Style::default().fg(Color::White).bg(Color::Blue))
        .alignment(Alignment::Right);

    frame.render_widget(text_area, main_layout[0]);
    frame.render_widget(status_text, status_layout[0]);
    frame.render_widget(memory_text, status_layout[1]);
    frame.render_widget(its_text, status_layout[2]);
}
