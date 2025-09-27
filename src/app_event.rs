use serde::Serialize;
use std::time::Duration;

#[derive(Clone, Debug, Serialize)]
pub struct LatencyRow {
    pub label: String,
    pub last_ms: f64,
    pub average_ms: f64,
    pub level: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryRow {
    pub label: String,
    pub level: u8,
    pub current_total_mb: f64,
    pub peak_total_mb: f64,
    pub current_pool_mb: f64,
    pub peak_pool_mb: f64,
    pub current_kv_mb: f64,
    pub peak_kv_mb: f64,
    pub current_kv_cache_mb: f64,
    pub peak_kv_cache_mb: f64,
    pub absolute_pool_mb: f64,
    pub absolute_kv_mb: f64,
    pub absolute_kv_cache_mb: f64,
    pub show_absolute: bool,
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
    MemoryUpdate(Vec<MemoryRow>),
    LatencyUpdate(Vec<LatencyRow>),
}
