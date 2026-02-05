//! Component modules.

pub mod chat;
pub mod common;
pub mod input;
pub mod sidebar;

// Re-exports for convenience
pub use chat::ChatView;
pub use input::{ChatInput, TextInput, register_input_bindings};
pub use sidebar::Sidebar;
