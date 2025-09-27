use std::time::Duration;

#[derive(Clone, Debug)]
pub struct LatencyRow {
    pub label: String,
    pub last_ms: f64,
    pub average_ms: f64,
    pub level: u8,
}

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
    LatencyUpdate(Vec<LatencyRow>),
}
