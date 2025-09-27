pub enum AppEvent {
    Token(String, f64),
    TokenCount(usize),
    StatusUpdate(String),
    MemoryUpdate(String),
}
