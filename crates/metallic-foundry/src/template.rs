use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::MetalError;

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A wrapper around minijinja for rendering chat templates.
pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    template_str: Arc<str>,
}

impl ChatTemplate {
    pub fn new(template: &str) -> Self {
        let mut env = minijinja::Environment::new();
        env.set_keep_trailing_newline(true);
        // Add common filters if needed. minijinja has most built-in.

        Self {
            env,
            template_str: Arc::from(template),
        }
    }

    pub fn render(
        &self,
        messages: &[Message],
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
    ) -> Result<String, MetalError> {
        let ctx = minijinja::context! {
            messages => messages,
            bos_token => bos_token,
            eos_token => eos_token,
            add_generation_prompt => add_generation_prompt,
        };

        // We use render_str for now as templates are dynamic from GGUF.
        // If we want to optimize, we could compile and cache it.
        self.env
            .render_str(&self.template_str, ctx)
            .map_err(|e| MetalError::InvalidOperation(format!("Template rendering failed: {}", e)))
    }
}

impl Clone for ChatTemplate {
    fn clone(&self) -> Self {
        Self::new(&self.template_str)
    }
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").field("template_str", &self.template_str).finish()
    }
}
