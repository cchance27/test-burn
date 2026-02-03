use rustc_hash::FxHashMap;

use crate::{
    error::MetalError, template::Message, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct FormatChatOp {
    model_id: Option<String>,
    input_var: String,
    output_var: String,
    add_generation_prompt: bool,
    mode: Option<String>,
    // Count of messages seen across turns for this op instance. Used to support both:
    // - full-history inputs (render only the suffix), and
    // - delta-only inputs (render all provided messages).
    total_message_count_seen: usize,
}

impl FormatChatOp {
    pub(crate) fn new(spec: crate::workflow::spec::FormatChatSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input_var: spec.input,
            output_var: spec.output,
            add_generation_prompt: spec.add_generation_prompt,
            mode: spec.mode,
            total_message_count_seen: 0,
        }
    }
}

fn select_messages_to_render<'a>(
    mode: &str,
    messages: &'a [Message],
    total_message_count_seen: &mut usize,
) -> Result<&'a [Message], MetalError> {
    match mode {
        "full" => {
            *total_message_count_seen = messages.len();
            Ok(messages)
        }
        "delta" => {
            // "delta" must support two usage patterns:
            // 1) Full-history input that grows over time (render only the suffix).
            // 2) Delta-only input (caller supplies only new messages each turn).
            //
            // We distinguish them defensively by looking at whether the input length is <= the
            // number of messages we've already seen. Full-history should monotonically increase.
            if *total_message_count_seen == 0 {
                *total_message_count_seen = messages.len();
                return Ok(messages);
            }

            if messages.len() <= *total_message_count_seen {
                // Treat as delta-only input: render all provided messages and advance the total.
                *total_message_count_seen = total_message_count_seen.saturating_add(messages.len());
                return Ok(messages);
            }

            // Full-history input grew: render only the new suffix.
            let start = *total_message_count_seen;
            *total_message_count_seen = messages.len();
            Ok(&messages[start..])
        }
        other => Err(MetalError::InvalidOperation(format!(
            "format_chat unsupported mode '{other}' (expected 'full' or 'delta')"
        ))),
    }
}

fn map_message(map: &FxHashMap<String, Value>) -> Result<Message, MetalError> {
    let role = map
        .get("role")
        .and_then(|v| v.as_text())
        .ok_or_else(|| MetalError::InvalidOperation("format_chat: message map missing 'role'".into()))?;
    let content = map
        .get("content")
        .and_then(|v| v.as_text())
        .ok_or_else(|| MetalError::InvalidOperation("format_chat: message map missing 'content'".into()))?;
    Ok(Message {
        role: role.to_string(),
        content: content.to_string(),
    })
}

fn format_qwen_messages(messages: &[Message], add_generation_prompt: bool) -> String {
    // Match Qwen2/Qwen2.5 canonical format expected by many instruct fine-tunes.
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

impl WorkflowOp for FormatChatOp {
    fn memoize_spec(&self) -> Option<crate::workflow::ops::MemoizeSpec> {
        Some(crate::workflow::ops::MemoizeSpec {
            inputs: vec![self.input_var.clone()],
            outputs: vec![self.output_var.clone()],
            // In TUI, message history changes every turn; this cache is unlikely to hit and would
            // retain large strings. Keep it off for now.
            disable_in_tui: true,
        })
    }

    fn begin_run(&mut self, _ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        // Keep total_message_count_seen across turns; it is the basis for delta formatting.
        Ok(())
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;
        let tokenizer = model.tokenizer()?;

        let input = ctx
            .values
            .get(&self.input_var)
            .ok_or_else(|| MetalError::InvalidOperation(format!("format_chat: missing input '{}'", self.input_var)))?;
        let arr = input
            .as_array()
            .ok_or_else(|| MetalError::InvalidOperation("format_chat: input must be an array of messages".into()))?;

        let mut messages: Vec<Message> = Vec::with_capacity(arr.len().min(64));
        for v in arr {
            match v {
                Value::Map(map) => messages.push(map_message(map)?),
                _ => {
                    return Err(MetalError::InvalidOperation(
                        "format_chat: each message must be a map with {role, content}".into(),
                    ));
                }
            }
        }

        let mode = self.mode.as_deref().unwrap_or("full");
        let messages_to_render = select_messages_to_render(mode, &messages, &mut self.total_message_count_seen)?;

        // If the tokenizer provides a chat template, use it. Otherwise fall back to Qwen formatting
        // when Qwen chat tokens are present, else concatenate contents.
        let formatted = if let Some(template) = tokenizer.chat_template() {
            let bos_token = tokenizer
                .special_tokens()
                .bos_token_id
                .and_then(|id| tokenizer.decode_lossless(&[id]).ok());
            let eos_token = tokenizer
                .special_tokens()
                .eos_token_id
                .and_then(|id| tokenizer.decode_lossless(&[id]).ok());
            template.render(
                messages_to_render,
                bos_token.as_deref(),
                eos_token.as_deref(),
                self.add_generation_prompt,
            )?
        } else if tokenizer.has_token("<|im_start|>") && tokenizer.has_token("<|im_end|>") {
            format_qwen_messages(messages_to_render, self.add_generation_prompt)
        } else {
            messages_to_render.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
        };

        if std::env::var("METALLIC_DEBUG_CHAT_TEMPLATE").is_ok() {
            eprintln!(
                "--- DEBUG: Chat Template Render ---\n{}\n-----------------------------------",
                formatted
            );
        }

        ctx.values.insert(self.output_var.clone(), Value::Text(formatted.into()));
        Ok(WorkflowOpOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str) -> Message {
        Message {
            role: role.to_string(),
            content: "x".to_string(),
        }
    }

    #[test]
    fn format_chat_delta_supports_delta_only_inputs_across_turns() {
        let mut seen = 0usize;

        // First turn: full history (system + user)
        let m1 = vec![msg("system"), msg("user")];
        let r1 = select_messages_to_render("delta", &m1, &mut seen).unwrap();
        assert_eq!(r1.len(), 2);
        assert_eq!(seen, 2);

        // Second turn: delta-only input (user only)
        let m2 = vec![msg("user")];
        let r2 = select_messages_to_render("delta", &m2, &mut seen).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(seen, 3);

        // Third turn: delta-only input again (user only) should still render 1 message (not empty).
        let m3 = vec![msg("user")];
        let r3 = select_messages_to_render("delta", &m3, &mut seen).unwrap();
        assert_eq!(r3.len(), 1);
        assert_eq!(seen, 4);
    }

    #[test]
    fn format_chat_delta_supports_full_history_inputs_by_rendering_suffix() {
        let mut seen = 0usize;

        // First call: full history.
        let m1 = vec![msg("system"), msg("user")];
        let r1 = select_messages_to_render("delta", &m1, &mut seen).unwrap();
        assert_eq!(r1.len(), 2);
        assert_eq!(seen, 2);

        // Next call: full history grew by one.
        let m2 = vec![msg("system"), msg("user"), msg("user")];
        let r2 = select_messages_to_render("delta", &m2, &mut seen).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(seen, 3);
    }
}
