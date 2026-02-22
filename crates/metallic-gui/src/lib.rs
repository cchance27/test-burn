//! Metallic GUI - A modern AI chat interface built with GPUI
//!
//! This crate provides a GPU-accelerated chat interface for AI conversations,
//! featuring a sidebar for conversation history and a main chat view.

pub mod app;
pub mod components;
pub mod db;
pub mod inference;
pub mod state;
pub mod theme;
pub mod types;

pub use app::ChatApp;
pub use types::{ChatMessage, Conversation, MessageRole};

// Global application actions
gpui::actions!(app, [Quit]);
