use metallic_foundry::template::{ChatTemplate, Message};

#[test]
fn chat_template_render_accepts_missing_bos_eos_tokens() {
    let t = ChatTemplate::new("{{ bos_token }}|{{ eos_token }}|{{ messages[0].role }}");
    let messages = vec![Message {
        role: "user".to_string(),
        content: "hi".to_string(),
    }];
    let out = t.render(&messages, None, None, false).expect("render");
    assert_eq!(out, "||user");
}
