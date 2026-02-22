use metallic_foundry::template::{ChatTemplate, Message};

#[test]
fn chat_template_render_fails_when_required_tokens_are_missing() {
    let t = ChatTemplate::new("{{ bos_token }}|{{ eos_token }}|{{ messages[0].role }}");
    let messages = vec![Message {
        role: "user".to_string(),
        content: "hi".to_string(),
    }];
    let err = t.render(&messages, None, None, false, None).expect_err("render should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("requires 'bos_token'") || msg.contains("requires 'eos_token'"),
        "unexpected error: {msg}"
    );
}

#[test]
fn chat_template_render_accepts_missing_bos_eos_when_unused() {
    let t = ChatTemplate::new("{{ messages[0].role }}");
    let messages = vec![Message {
        role: "user".to_string(),
        content: "hi".to_string(),
    }];
    let out = t.render(&messages, None, None, false, None).expect("render");
    assert_eq!(out, "user");
}
