use std::time::Duration;

pub enum AppEvent {
    Token {
        text: String,
        tokens_per_second: f64,
        prompt_processing: Duration,
        generation: Duration,
    },
    TokenCount(usize),
    StatusUpdate(String),
    MemoryUpdate(String),
}
