use metallic_env::{FoundryEnvVar, SYSTEM_PROMPT, is_set};

use super::BPETokenizer;
use crate::{MetalError, template::Message};

impl BPETokenizer {
    fn system_prompt_from_env() -> Option<String> {
        SYSTEM_PROMPT
            .get()
            .ok()
            .flatten()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[inline]
    fn debug_dump_chat_template_render(kind: &str, formatted: &str) {
        if is_set(FoundryEnvVar::DebugChatTemplate) {
            eprintln!(
                "[metallic][debug] chat_template_render kind={} chars={}\n{}",
                kind,
                formatted.chars().count(),
                formatted
            );
        }
    }

    fn format_im_marker_messages(messages: &[Message], add_generation_prompt: bool) -> String {
        let mut cap = 0usize;
        for m in messages {
            cap = cap.saturating_add(m.role.len()).saturating_add(m.content.len()).saturating_add(32);
        }
        let mut s = String::with_capacity(cap.saturating_add(32));
        for m in messages {
            s.push_str("<|im_start|>");
            s.push_str(m.role.as_str());
            s.push('\n');
            s.push_str(m.content.as_str());
            s.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            s.push_str("<|im_start|>assistant\n");
        }
        s
    }

    /// Canonical chat-message rendering entrypoint used by both tokenizer helpers and workflow `format_chat`.
    pub fn render_chat_messages(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<(String, &'static str), MetalError> {
        if let Some(template) = &self.chat_template {
            let bos_token = self
                .special_tokens
                .bos_token_id
                .and_then(|id| self.vocab.get(&id).map(|s| s.as_ref()));
            let eos_token = self
                .special_tokens
                .eos_token_id
                .and_then(|id| self.vocab.get(&id).map(|s| s.as_ref()));

            let formatted = template.render(messages, bos_token, eos_token, add_generation_prompt, template_kwargs)?;
            Self::debug_dump_chat_template_render("messages", &formatted);
            return Ok((formatted, "chat_template"));
        }

        let has_im_markers = self.has_token("<|im_start|>") && self.has_token("<|im_end|>");
        if has_im_markers {
            let formatted = Self::format_im_marker_messages(messages, add_generation_prompt);
            Self::debug_dump_chat_template_render("im_marker_fallback", &formatted);
            return Ok((formatted, "im_marker_fallback"));
        }

        Ok((
            messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n"),
            "plain_join",
        ))
    }

    pub fn set_chat_template(&mut self, template: String) {
        self.chat_template = Some(crate::template::ChatTemplate::new(&template));
    }

    /// Format a single-turn chat prompt using the dynamic template.
    ///
    /// If the tokenizer does not have a chat template, it returns the prompt as-is.
    pub fn format_single_turn_chat_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, MetalError> {
        // Fast-path: if the prompt is already chat-formatted, do not wrap it again.
        if prompt.contains("<|im_start|>") {
            return Ok(prompt.to_string());
        }

        if self.chat_template.is_some() {
            let mut messages = Vec::with_capacity(2);
            if let Some(system) = Self::system_prompt_from_env() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: system,
                });
            }
            messages.push(Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            });
            let (formatted, _path) = self.render_chat_messages(&messages, true, template_kwargs)?;
            return Ok(formatted);
        }

        // Fallback: if the tokenizer doesn't provide a template, pass raw prompt through.
        Ok(prompt.to_string())
    }

    pub fn format_single_turn_chat_prompt(&self, prompt: &str) -> Result<String, MetalError> {
        self.format_single_turn_chat_prompt_with_kwargs(prompt, None)
    }

    pub fn encode_single_turn_chat_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Vec<u32>, MetalError> {
        let formatted = self.format_single_turn_chat_prompt_with_kwargs(prompt, template_kwargs)?;
        self.encode(&formatted)
    }

    pub fn encode_single_turn_chat_prompt(&self, prompt: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_single_turn_chat_prompt_with_kwargs(prompt, None)
    }

    /// Format a continuation chat prompt using the dynamic template.
    ///
    /// This adds the user prompt and opens the assistant turn.
    pub fn format_chat_continuation_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, MetalError> {
        if self.chat_template.is_none() {
            return Ok(prompt.to_string());
        }

        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let (formatted, _path) = self.render_chat_messages(&messages, true, template_kwargs)?;
        Ok(formatted)
    }

    pub fn format_chat_continuation_prompt(&self, prompt: &str) -> Result<String, MetalError> {
        self.format_chat_continuation_prompt_with_kwargs(prompt, None)
    }

    pub fn encode_chat_continuation_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Vec<u32>, MetalError> {
        let formatted = self.format_chat_continuation_prompt_with_kwargs(prompt, template_kwargs)?;
        self.encode(&formatted)
    }

    pub fn encode_chat_continuation_prompt(&self, prompt: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_chat_continuation_prompt_with_kwargs(prompt, None)
    }
}
