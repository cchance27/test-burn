//! Core data types for the chat interface.

use chrono::{DateTime, Local};

/// Role of the message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Performance metrics for a single inference generation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InferencePerf {
    /// Number of tokens emitted.
    pub tokens: usize,
    /// Total wall-clock duration in milliseconds.
    pub wall_ms: u128,
    /// Wall-clock tokens per second.
    pub wall_tok_per_sec: f64,
    /// Pure decode tokens per second (excluding prefill).
    pub decode_tok_per_sec: f64,
    /// Latency to first token in milliseconds.
    pub first_token_ms: u128,
    /// Time spent in prefill in milliseconds.
    pub prefill_ms: u128,
    /// One-time setup latency in milliseconds (locks/setup/bind checks) before decode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub setup_ms: Option<u128>,
    /// Time spent preparing prompt/workflow before first decode launch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_prep_ms: Option<u128>,
    /// Decode kernel/sample time for the first emitted token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_decode_ms: Option<u128>,
    /// Remaining time-to-first-token after setup+prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_wait_ms: Option<u128>,
    /// Number of tokens prefilled.
    pub prefill_tokens: usize,
    /// Prefill tokens per second.
    pub prefill_tok_per_sec: f64,
    /// Model context usage after this generation (prompt + emitted tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<usize>,
}

/// Metadata captured during message generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MessageMetadata {
    /// Filename of the model used.
    pub model_filename: String,
    /// Snapshotted generation parameters (temperature, top_p, etc.).
    pub params: std::collections::HashMap<String, serde_json::Value>,
    /// Performance metrics for this specific generation.
    pub perf: InferencePerf,
}

/// A single chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: u64,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Local>,
    pub metadata: Option<MessageMetadata>,
}

impl ChatMessage {
    pub fn new(id: u64, role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            id,
            role,
            content: content.into(),
            timestamp: Local::now(),
            metadata: None,
        }
    }

    pub fn user(id: u64, content: impl Into<String>) -> Self {
        Self::new(id, MessageRole::User, content)
    }

    pub fn assistant(id: u64, content: impl Into<String>) -> Self {
        Self::new(id, MessageRole::Assistant, content)
    }
}

/// A conversation containing multiple messages.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: u64,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: DateTime<Local>,
    pub updated_at: DateTime<Local>,
}

impl Conversation {
    pub fn new(id: u64, title: impl Into<String>) -> Self {
        let now = Local::now();
        Self {
            id,
            title: title.into(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.updated_at = Local::now();
        self.messages.push(message);
    }

    /// Generate a preview of the last message (for sidebar display).
    pub fn preview(&self) -> &str {
        self.messages.last().map(|m| m.content.as_str()).unwrap_or("No messages yet")
    }
}

/// Create a set of placeholder conversations for demo purposes.
pub fn placeholder_conversations() -> Vec<Conversation> {
    let mut convos = vec![
        Conversation::new(1, "How to build a compiler"),
        Conversation::new(2, "Rust async patterns"),
        Conversation::new(3, "Metal shader optimization"),
        Conversation::new(4, "GPUI best practices"),
        Conversation::new(5, "Machine learning basics"),
    ];

    // Add placeholder messages to first conversation
    convos[0].messages = vec![
        ChatMessage::user(1, "Can you explain how compilers work?"),
        ChatMessage::assistant(
            2,
            "Compilers transform source code into machine code through several phases: \
             lexical analysis, parsing, semantic analysis, optimization, and code generation. \
             Each phase builds upon the previous one to produce efficient executable code.",
        ),
        ChatMessage::user(3, "What's the difference between a compiler and an interpreter?"),
        ChatMessage::assistant(
            4,
            "A compiler translates the entire program before execution, producing standalone \
             executables. An interpreter executes code line-by-line at runtime. Compilers \
             typically produce faster programs, while interpreters offer more flexibility \
             and easier debugging.",
        ),
    ];

    // Add some messages to second conversation
    convos[1].messages = vec![
        ChatMessage::user(1, "What are the main async patterns in Rust?"),
        ChatMessage::assistant(
            2,
            "Rust's async ecosystem centers around futures, async/await syntax, and executors. \
             Key patterns include select! for racing futures, join! for concurrent execution, \
             and channels for communication between tasks.",
        ),
    ];

    convos
}
