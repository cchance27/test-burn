use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

use crate::{
    error::MetalError,
    template::Message,
    workflow::{
        Value,
        ops::{WorkflowOp, WorkflowOpOutcome},
        runner::WorkflowExecutionContext,
    },
};

pub(crate) struct FormatChatOp {
    model_id: Option<String>,
    input_var: String,
    system_prompt_var: Option<String>,
    output_var: String,
    add_generation_prompt: bool,
    mode: Option<String>,
    default_state: ConversationFormatState,
    conversation_states: FxHashMap<String, ConversationFormatState>,
}

#[derive(Default)]
struct ConversationFormatState {
    total_message_count_seen: usize,
    system_prompt_rendered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderSelectionKind {
    Full,
    DeltaFirstTurn,
    DeltaFullGrowthSuffix,
    DeltaOnlyInput,
}

#[derive(Clone, Copy, Debug)]
struct RenderSelection<'a> {
    messages: &'a [Message],
    kind: RenderSelectionKind,
    skipped_assistant_prefix: usize,
}

const KV_PREFIX_KEY_VAR: &str = "_internal.kv_prefix_key";
const KV_PREFIX_BASE_KEY_VAR: &str = "_internal.kv_prefix_base_key";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SystemPrefixSource {
    None,
    AlreadyPresent,
    FullHistory,
    WorkflowPrompt,
}

impl ConversationFormatState {
    fn reset(&mut self) {
        self.total_message_count_seen = 0;
        self.system_prompt_rendered = false;
    }
}

impl FormatChatOp {
    pub(crate) fn new(spec: crate::workflow::spec::FormatChatSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input_var: spec.input,
            system_prompt_var: spec.system_prompt,
            output_var: spec.output,
            add_generation_prompt: spec.add_generation_prompt,
            mode: spec.mode,
            default_state: ConversationFormatState::default(),
            conversation_states: FxHashMap::default(),
        }
    }

    fn state_for_key<'a>(&'a mut self, key: Option<&str>) -> &'a mut ConversationFormatState {
        if let Some(key) = key {
            return self.conversation_states.entry(key.to_string()).or_default();
        }
        &mut self.default_state
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

fn select_messages_to_render<'a>(
    mode: &str,
    messages: &'a [Message],
    total_message_count_seen: &mut usize,
) -> Result<RenderSelection<'a>, MetalError> {
    match mode {
        "full" => {
            *total_message_count_seen = messages.len();
            Ok(RenderSelection {
                messages,
                kind: RenderSelectionKind::Full,
                skipped_assistant_prefix: 0,
            })
        }
        "delta" => {
            // "delta" supports:
            // 1) Full-history input that grows each turn (render only suffix), and
            // 2) Delta-only input (caller provides only new turns).
            if *total_message_count_seen == 0 {
                *total_message_count_seen = messages.len();
                return Ok(RenderSelection {
                    messages,
                    kind: RenderSelectionKind::DeltaFirstTurn,
                    skipped_assistant_prefix: 0,
                });
            }

            if messages.len() <= *total_message_count_seen {
                *total_message_count_seen = (*total_message_count_seen).saturating_add(messages.len());
                return Ok(RenderSelection {
                    messages,
                    kind: RenderSelectionKind::DeltaOnlyInput,
                    skipped_assistant_prefix: 0,
                });
            }

            let mut start = *total_message_count_seen;
            // Generated assistant turns are already in-session via decode path; don't replay them.
            while start < messages.len() && messages[start].role == "assistant" {
                start += 1;
            }
            let skipped_assistant_prefix = start.saturating_sub(*total_message_count_seen);
            *total_message_count_seen = messages.len();
            Ok(RenderSelection {
                messages: &messages[start..],
                kind: RenderSelectionKind::DeltaFullGrowthSuffix,
                skipped_assistant_prefix,
            })
        }
        other => Err(MetalError::InvalidOperation(format!(
            "format_chat unsupported mode '{other}' (expected 'full' or 'delta')"
        ))),
    }
}

fn ensure_system_prefix(
    messages_to_render: &mut Vec<Message>,
    full_messages: &[Message],
    workflow_system_prompt: Option<&str>,
) -> SystemPrefixSource {
    if messages_to_render.first().map(|m| m.role.as_str()) == Some("system") {
        return SystemPrefixSource::AlreadyPresent;
    }

    if let Some(system_msg) = full_messages.first().filter(|m| m.role == "system") {
        messages_to_render.insert(0, system_msg.clone());
        return SystemPrefixSource::FullHistory;
    }

    if let Some(system_prompt) = workflow_system_prompt {
        messages_to_render.insert(
            0,
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
        );
        return SystemPrefixSource::WorkflowPrompt;
    }

    SystemPrefixSource::None
}

fn conversation_key_from_ctx(ctx: &WorkflowExecutionContext<'_>) -> Option<String> {
    let value = ctx.values.get("conversation_id")?;
    if let Some(v) = value.as_text() {
        return Some(v.to_string());
    }
    if let Some(v) = value.as_u32() {
        return Some(v.to_string());
    }
    value.as_usize().map(|v| v.to_string())
}

fn format_qwen_messages(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut cap = 0usize;
    for m in messages {
        cap = cap
            .saturating_add(m.role.len())
            .saturating_add(m.content.len())
            .saturating_add(32);
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

fn hash_message_key(tag: &str, add_generation_prompt: bool, messages: &[(&str, &str)]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    "kv-prefix-v2".hash(&mut hasher);
    tag.hash(&mut hasher);
    add_generation_prompt.hash(&mut hasher);
    let mut bytes = 0usize;
    for (role, content) in messages {
        role.hash(&mut hasher);
        content.hash(&mut hasher);
        bytes = bytes.saturating_add(role.len()).saturating_add(content.len());
    }
    format!("{tag}:{:016x}:{}:{}", hasher.finish(), messages.len(), bytes)
}

fn compute_kv_prefix_keys(
    mode: &str,
    selection_kind: RenderSelectionKind,
    messages_to_render: &[Message],
    full_messages: &[Message],
    workflow_system_prompt: Option<&str>,
    add_generation_prompt: bool,
) -> (Option<String>, Option<String>) {
    if mode != "delta" || !matches!(selection_kind, RenderSelectionKind::DeltaFirstTurn) {
        return (None, None);
    }

    let system_content = messages_to_render
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .or_else(|| {
            full_messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.as_str())
        })
        .or(workflow_system_prompt);
    let Some(system_content) = system_content else {
        let user_content = messages_to_render
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .or_else(|| {
                full_messages
                    .iter()
                    .find(|m| m.role == "user")
                    .map(|m| m.content.as_str())
            });
        if let Some(user_content) = user_content {
            let primary = hash_message_key("user", add_generation_prompt, &[("user", user_content)]);
            return (Some(primary), None);
        }
        return (None, None);
    };

    let user_content = messages_to_render
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .or_else(|| {
            full_messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.as_str())
        });
    let (primary, base) = if let Some(user_content) = user_content {
        let primary = hash_message_key(
            "system_user",
            add_generation_prompt,
            &[("system", system_content), ("user", user_content)],
        );
        // Base key intentionally uses an empty user-content scaffold so warmup and first-turn
        // requests share a stable prefix cache key without depending on user text.
        let base = hash_message_key(
            "system_user",
            add_generation_prompt,
            &[("system", system_content), ("user", "")],
        );
        (primary, base)
    } else {
        let base = hash_message_key("system", false, &[("system", system_content)]);
        (base.clone(), base)
    };

    (Some(primary), Some(base))
}

fn render_messages_with_tokenizer(
    tokenizer: &crate::BPETokenizer,
    messages: &[Message],
    add_generation_prompt: bool,
    prefer_qwen_delta_canonical: bool,
) -> Result<(String, &'static str), MetalError> {
    let has_qwen_markers = tokenizer.has_token("<|im_start|>") && tokenizer.has_token("<|im_end|>");
    if prefer_qwen_delta_canonical {
        return Ok((format_qwen_messages(messages, add_generation_prompt), "qwen_delta_canonical"));
    }

    if let Some(template) = tokenizer.chat_template() {
        let bos_token = tokenizer
            .special_tokens()
            .bos_token_id
            .and_then(|id| tokenizer.decode_lossless(&[id]).ok());
        let eos_token = tokenizer
            .special_tokens()
            .eos_token_id
            .and_then(|id| tokenizer.decode_lossless(&[id]).ok());
        return Ok((
            template.render(messages, bos_token.as_deref(), eos_token.as_deref(), add_generation_prompt)?,
            "chat_template",
        ));
    }

    if has_qwen_markers {
        return Ok((format_qwen_messages(messages, add_generation_prompt), "qwen_fallback"));
    }

    Ok((
        messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        "plain_join",
    ))
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
        // Keep per-conversation state across turns.
        Ok(())
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let mode = self.mode.clone().unwrap_or_else(|| "full".to_string());
        if mode != "full" && mode != "delta" {
            return Err(MetalError::InvalidOperation(format!(
                "format_chat unsupported mode '{mode}' (expected 'full' or 'delta')"
            )));
        }
        ctx.values.remove(KV_PREFIX_KEY_VAR);
        ctx.values.remove(KV_PREFIX_BASE_KEY_VAR);

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

        let conversation_key = conversation_key_from_ctx(ctx);
        let model_id = self.model_id.clone();
        let system_prompt_var = self.system_prompt_var.clone();
        let workflow_system_prompt = system_prompt_var
            .as_deref()
            .and_then(|sys_var| ctx.values.get(sys_var))
            .and_then(|v| v.as_text());

        let (mut total_message_count_seen, mut system_prompt_rendered) = {
            let state = self.state_for_key(conversation_key.as_deref());
            (state.total_message_count_seen, state.system_prompt_rendered)
        };

        let seen_before = total_message_count_seen;
        let system_before = system_prompt_rendered;
        let selection = select_messages_to_render(mode.as_str(), &messages, &mut total_message_count_seen)?;
        let messages_slice = selection.messages;
        let mut messages_to_render = messages_slice.to_vec();
        let system_prefix_source = if !system_prompt_rendered {
            ensure_system_prefix(&mut messages_to_render, &messages, workflow_system_prompt)
        } else {
            SystemPrefixSource::None
        };
        let system_injected = matches!(
            system_prefix_source,
            SystemPrefixSource::FullHistory | SystemPrefixSource::WorkflowPrompt
        );
        if system_injected {
            tracing::debug!(
                target: "metallic_foundry::workflow::ops::format_chat",
                conversation_id = conversation_key.as_deref().unwrap_or("<default>"),
                mode = mode.as_str(),
                source = ?system_prefix_source,
                "format_chat injected system prefix into rendered delta"
            );
        }
        if messages_to_render.first().map(|m| m.role.as_str()) == Some("system") {
            system_prompt_rendered = true;
        }

        let model = ctx.resolve_model(model_id.as_deref())?;
        let tokenizer = model.tokenizer()?;
        let add_generation_prompt = self.add_generation_prompt && !messages_slice.is_empty();
        let has_qwen_markers = tokenizer.has_token("<|im_start|>") && tokenizer.has_token("<|im_end|>");
        let use_qwen_delta_canonical = mode == "delta"
            && system_prompt_rendered
            && !messages_to_render.is_empty()
            && messages_to_render.first().map(|m| m.role.as_str()) != Some("system")
            && has_qwen_markers;

        // For Qwen-like tokenizers in delta mode, render suffix with canonical format to avoid
        // template-side default system insertion when suffix starts with a user turn.
        let (formatted, render_path) =
            render_messages_with_tokenizer(&tokenizer, &messages_to_render, add_generation_prompt, use_qwen_delta_canonical)?;

        let (kv_prefix_key, kv_prefix_base_key) = compute_kv_prefix_keys(
            mode.as_str(),
            selection.kind,
            &messages_to_render,
            &messages,
            workflow_system_prompt,
            add_generation_prompt,
        );

        if let Some(key) = kv_prefix_key.as_ref() {
            ctx.values.insert(KV_PREFIX_KEY_VAR.to_string(), Value::Text(key.as_str().into()));
        }
        if let Some(key) = kv_prefix_base_key.as_ref() {
            ctx.values.insert(KV_PREFIX_BASE_KEY_VAR.to_string(), Value::Text(key.as_str().into()));
        }

        {
            let state = self.state_for_key(conversation_key.as_deref());
            state.total_message_count_seen = total_message_count_seen;
            state.system_prompt_rendered = system_prompt_rendered;
        }

        tracing::debug!(
            target: "metallic_foundry::workflow::ops::format_chat",
            conversation_id = conversation_key.as_deref().unwrap_or("<default>"),
            mode = mode.as_str(),
            input_messages = messages.len(),
            selected_messages = messages_slice.len(),
            rendered_messages = messages_to_render.len(),
            seen_before,
            seen_after = total_message_count_seen,
            selection_kind = ?selection.kind,
            skipped_assistant_prefix = selection.skipped_assistant_prefix,
            system_before,
            system_after = system_prompt_rendered,
            system_prefix = ?system_prefix_source,
            add_generation_prompt,
            render_path,
            rendered_bytes = formatted.len(),
            kv_prefix_key = kv_prefix_key.as_deref().unwrap_or("<none>"),
            kv_prefix_base_key = kv_prefix_base_key.as_deref().unwrap_or("<none>"),
            "format_chat path selection"
        );
        tracing::trace!(
            target: "metallic_foundry::workflow::ops::format_chat",
            conversation_id = conversation_key.as_deref().unwrap_or("<default>"),
            rendered = %formatted,
            "[metallic][chat_template_render]"
        );

        ctx.values.insert(self.output_var.clone(), Value::Text(formatted.into()));
        Ok(WorkflowOpOutcome::Continue)
    }

    fn reset(&mut self) {
        tracing::debug!(
            target: "metallic_foundry::workflow::ops::format_chat",
            cleared_conversation_states = self.conversation_states.len(),
            "format_chat reset state"
        );
        self.default_state.reset();
        self.conversation_states.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg_with(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn ensure_system_prefix_uses_system_from_full_history() {
        let full = vec![
            msg_with("system", "Alloy system"),
            msg_with("user", "u1"),
            msg_with("assistant", "a1"),
            msg_with("user", "u2"),
        ];
        let mut render = vec![msg_with("user", "u2")];

        let source = ensure_system_prefix(&mut render, &full, Some("fallback"));

        assert_eq!(render.len(), 2);
        assert_eq!(render[0].role, "system");
        assert_eq!(render[0].content, "Alloy system");
        assert_eq!(render[1].role, "user");
        assert_eq!(source, SystemPrefixSource::FullHistory);
    }

    #[test]
    fn ensure_system_prefix_falls_back_to_workflow_prompt() {
        let full = vec![msg_with("user", "u1")];
        let mut render = vec![msg_with("user", "u2")];

        let source = ensure_system_prefix(&mut render, &full, Some("workflow system"));

        assert_eq!(render.len(), 2);
        assert_eq!(render[0].role, "system");
        assert_eq!(render[0].content, "workflow system");
        assert_eq!(source, SystemPrefixSource::WorkflowPrompt);
    }

    #[test]
    fn ensure_system_prefix_noop_when_already_present() {
        let full = vec![msg_with("system", "Alloy system"), msg_with("user", "u1")];
        let mut render = vec![msg_with("system", "Alloy system"), msg_with("user", "u2")];

        let source = ensure_system_prefix(&mut render, &full, Some("workflow system"));

        assert_eq!(render.len(), 2);
        assert_eq!(render[0].role, "system");
        assert_eq!(render[0].content, "Alloy system");
        assert_eq!(source, SystemPrefixSource::AlreadyPresent);
    }

    #[test]
    fn select_messages_delta_first_call_renders_all() {
        let mut seen = 0usize;
        let m1 = vec![msg_with("system", "s"), msg_with("user", "u1")];
        let out = select_messages_to_render("delta", &m1, &mut seen).unwrap();
        assert_eq!(out.messages.len(), 2);
        assert_eq!(seen, 2);
        assert_eq!(out.kind, RenderSelectionKind::DeltaFirstTurn);
    }

    #[test]
    fn select_messages_delta_full_history_growth_skips_assistant_prefix() {
        let mut seen = 2usize;
        let m = vec![
            msg_with("system", "s"),
            msg_with("user", "u1"),
            msg_with("assistant", "a1"),
            msg_with("user", "u2"),
        ];
        let out = select_messages_to_render("delta", &m, &mut seen).unwrap();
        assert_eq!(out.messages.len(), 1);
        assert_eq!(out.messages[0].role, "user");
        assert_eq!(seen, 4);
        assert_eq!(out.kind, RenderSelectionKind::DeltaFullGrowthSuffix);
        assert_eq!(out.skipped_assistant_prefix, 1);
    }

    #[test]
    fn select_messages_delta_delta_only_input_renders_as_is() {
        let mut seen = 4usize;
        let m = vec![msg_with("user", "u3")];
        let out = select_messages_to_render("delta", &m, &mut seen).unwrap();
        assert_eq!(out.messages.len(), 1);
        assert_eq!(out.messages[0].role, "user");
        assert_eq!(seen, 5);
        assert_eq!(out.kind, RenderSelectionKind::DeltaOnlyInput);
    }

    #[test]
    fn compute_kv_prefix_keys_system_only_uses_stable_system_key() {
        let msgs = vec![msg_with("system", "Alloy system")];
        let (primary, base) = compute_kv_prefix_keys(
            "delta",
            RenderSelectionKind::DeltaFirstTurn,
            &msgs,
            &msgs,
            Some("Alloy system"),
            false,
        );
        let expected = hash_message_key("system", false, &[("system", "Alloy system")]);
        assert_eq!(primary.as_deref(), Some(expected.as_str()));
        assert_eq!(base.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn compute_kv_prefix_keys_system_user_primary_and_base() {
        let full = vec![msg_with("system", "Alloy system"), msg_with("user", "hello")];
        let render = full.clone();
        let (primary, base) = compute_kv_prefix_keys(
            "delta",
            RenderSelectionKind::DeltaFirstTurn,
            &render,
            &full,
            Some("Alloy system"),
            true,
        );
        let expected_primary =
            hash_message_key("system_user", true, &[("system", "Alloy system"), ("user", "hello")]);
        let expected_base =
            hash_message_key("system_user", true, &[("system", "Alloy system"), ("user", "")]);
        assert_eq!(primary.as_deref(), Some(expected_primary.as_str()));
        assert_eq!(base.as_deref(), Some(expected_base.as_str()));
    }
}
