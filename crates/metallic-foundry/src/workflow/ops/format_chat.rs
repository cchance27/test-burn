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
}

impl FormatChatOp {
    pub(crate) fn new(spec: crate::workflow::spec::FormatChatSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input_var: spec.input,
            output_var: spec.output,
            add_generation_prompt: spec.add_generation_prompt,
        }
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
            template.render(&messages, bos_token.as_deref(), eos_token.as_deref(), self.add_generation_prompt)?
        } else if tokenizer.has_token("<|im_start|>") && tokenizer.has_token("<|im_end|>") {
            format_qwen_messages(&messages, self.add_generation_prompt)
        } else {
            messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
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
