#![cfg(test)]

use std::{collections::BTreeSet, path::PathBuf};

use metallic_loader::ModelLoader;

use super::*;

fn msg_with(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: content.to_string(),
    }
}

fn char_level_tokenizer_with_template(template: &str, samples: &[&str]) -> crate::BPETokenizer {
    let mut chars = BTreeSet::new();
    for sample in samples {
        for ch in sample.chars() {
            chars.insert(ch);
        }
    }
    for ch in template.chars() {
        chars.insert(ch);
    }

    let mut vocab = FxHashMap::default();
    vocab.insert(0_u32, "?".to_string());
    let mut next_id = 1_u32;
    for ch in chars {
        vocab.insert(next_id, ch.to_string());
        next_id = next_id.saturating_add(1);
    }

    crate::BPETokenizer::new(
        vocab,
        Vec::new(),
        FxHashMap::default(),
        crate::tokenizer::SpecialTokens::default(),
        false,
        Some(template.to_string()),
    )
    .expect("tokenizer should build")
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
    let expected_primary = hash_message_key("system_user", true, &[("system", "Alloy system"), ("user", "hello")]);
    let expected_base = hash_message_key("system_user", true, &[("system", "Alloy system"), ("user", "")]);
    assert_eq!(primary.as_deref(), Some(expected_primary.as_str()));
    assert_eq!(base.as_deref(), Some(expected_base.as_str()));
}

#[test]
fn longest_common_prefix_len_tokenwise() {
    let lhs = vec![1_u32, 2, 3, 9];
    let rhs = vec![1_u32, 2, 4];
    assert_eq!(longest_common_prefix_len(&lhs, &rhs), 2);
}

fn format_llama_messages(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str("<|start_header_id|>");
        s.push_str(m.role.as_str());
        s.push_str("<|end_header_id|>\n\n");
        s.push_str(m.content.as_str());
        s.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    s
}

fn format_im_messages(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut s = String::new();
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

#[test]
fn template_lcp_delta_matches_im_marker_suffix_for_user_turn() {
    // Canonical `<|im_start|> ... <|im_end|>` chat-template shape.
    let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let full = vec![
        msg_with("system", "s"),
        msg_with("user", "u1"),
        msg_with("assistant", "a1"),
        msg_with("user", "u2"),
    ];
    let prev_render = format_im_messages(&full[..3], false);
    let curr_render = format_im_messages(&full, true);
    let expected_delta = format_im_messages(&full[3..], true);
    let tokenizer = char_level_tokenizer_with_template(template, &[&prev_render, &curr_render, &expected_delta]);

    let delta = compute_template_lcp_delta(&tokenizer, &full, 3, true, Some("conversation-1"), None)
        .expect("lcp delta should compute")
        .expect("lcp delta should exist");
    let expected_tokens = tokenizer.encode(&expected_delta).expect("expected tokens");
    let delta_tokens = tokenizer.encode(&delta).expect("delta tokens");
    assert_eq!(delta_tokens, expected_tokens);
}

#[test]
fn template_lcp_delta_matches_llama_suffix_for_user_turn() {
    // Canonical Llama-3-style chat template shape.
    let template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";
    let full = vec![
        msg_with("system", "s"),
        msg_with("user", "u1"),
        msg_with("assistant", "a1"),
        msg_with("user", "u2"),
    ];
    let prev_render = format_llama_messages(&full[..3], false);
    let curr_render = format_llama_messages(&full, true);
    let expected_delta = format_llama_messages(&full[3..], true);
    let tokenizer = char_level_tokenizer_with_template(template, &[&prev_render, &curr_render, &expected_delta]);

    let delta = compute_template_lcp_delta(&tokenizer, &full, 3, true, Some("conversation-llama"), None)
        .expect("lcp delta should compute")
        .expect("lcp delta should exist");
    let expected_tokens = tokenizer.encode(&expected_delta).expect("expected tokens");
    let delta_tokens = tokenizer.encode(&delta).expect("delta tokens");
    assert_eq!(delta_tokens, expected_tokens);
}

#[test]
fn render_messages_with_template_preserves_control_bos_token_text() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0_u32, "?".to_string());
    vocab.insert(1_u32, "<|begin_of_text|>".to_string());

    let mut token_types = FxHashMap::default();
    token_types.insert(1_u32, 3); // control token

    let tokenizer = crate::BPETokenizer::new(
        vocab,
        Vec::new(),
        token_types,
        crate::tokenizer::SpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: None,
            pad_token_id: None,
        },
        false,
        Some("{{ bos_token }}{{ messages[0].content }}".to_string()),
    )
    .expect("tokenizer should build");

    let messages = vec![msg_with("user", "hello")];
    let (rendered, path) = render_messages_with_tokenizer(&tokenizer, &messages, true, None).expect("render should work");
    assert_eq!(path, "chat_template");
    assert_eq!(rendered, "<|begin_of_text|>hello");
}

#[test]
fn render_messages_with_template_kwargs_controls_thinking_mode() {
    let tokenizer = crate::BPETokenizer::new(
        FxHashMap::default(),
        Vec::new(),
        FxHashMap::default(),
        crate::tokenizer::SpecialTokens::default(),
        false,
        Some("{% if enable_thinking %}/think{% else %}/no_think{% endif %}".to_string()),
    )
    .expect("tokenizer should build");

    let messages = vec![msg_with("user", "hello")];
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(false));
    let (rendered, path) = render_messages_with_tokenizer(&tokenizer, &messages, true, Some(&kwargs)).expect("render should work");
    assert_eq!(path, "chat_template");
    assert_eq!(rendered, "/no_think");
}

#[test]
fn template_lcp_delta_matches_real_im_marker_template_when_available() {
    let gguf_path = std::env::var("METALLIC_CHAT_TEMPLATE_GGUF_TEST_PATH").ok().map(PathBuf::from);
    let Some(gguf_path) = gguf_path else {
        return;
    };
    if !gguf_path.exists() {
        return;
    }

    let model_loaded = match ModelLoader::from_file(&gguf_path) {
        Ok(model) => model,
        Err(_) => return,
    };
    let tokenizer = match crate::BPETokenizer::from_metadata(model_loaded.metadata()) {
        Ok(tok) => tok,
        Err(_) => return,
    };
    if tokenizer.chat_template().is_none() {
        return;
    }

    let full = vec![
        msg_with("system", "Alloy system"),
        msg_with("user", "create a js hello world"),
        msg_with("assistant", "console.log('Hello World');"),
        msg_with("user", "now in rustlang"),
    ];
    let expected_delta = format_im_messages(&full[3..], true);
    let delta = compute_template_lcp_delta(&tokenizer, &full, 3, true, Some("conversation-1"), None)
        .expect("lcp delta should compute")
        .expect("lcp delta should exist");

    let expected_tokens = tokenizer.encode(&expected_delta).expect("expected tokens");
    let delta_tokens = tokenizer.encode(&delta).expect("delta tokens");
    assert_eq!(delta_tokens, expected_tokens);
}
