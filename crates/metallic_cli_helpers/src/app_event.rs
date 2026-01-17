use std::{sync::Arc, time::Duration};

use chrono::{DateTime, Utc};
use serde::Serialize;

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

#[derive(Clone, Debug, Serialize)]
pub struct StatsRow {
    pub label: String,
    pub value: String, // Can be numeric or a formatted string
    pub level: u8,
    pub description: String,
}

#[derive(Clone, Debug, Serialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
}

impl AlertLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Alert {
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

impl Alert {
    pub fn new(level: AlertLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            timestamp: Utc::now(),
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Info, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Warning, message)
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Error, message)
    }
}

pub enum AppEvent {
    Token {
        text: Arc<str>,
        /// Time spent in engine-side setup before the first prompt forward pass.
        ///
        /// This is primarily useful for performance breakdowns so that "Decode" can exclude
        /// one-time setup costs (buffer allocation, pipeline warmup, etc.). Engines that don't
        /// report this should set it to `None`.
        setup_duration: Option<Duration>,
        prompt_processing: Duration,
        iteration: Option<Duration>,
    },
    GenerationComplete {
        total_generation_time: Duration,
    },
    ModelLoadComplete(Duration),
    TokenizationComplete(Duration),
    TokenCount(usize),
    StatusUpdate(String),
    MemoryUpdate(Vec<MemoryRow>),
    LatencyUpdate(Vec<LatencyRow>),
    StatsUpdate(Vec<StatsRow>),
    Alert(Alert),
    LogMessage(String),
}
