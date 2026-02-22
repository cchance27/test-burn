#![cfg(test)]

use super::{ChatTemplate, Message};

#[test]
fn render_supports_strftime_and_generation_tags() {
    let template = "{% generation %}{%- set today = strftime_now(\"%d %B %Y\") -%}{% endgeneration %}Today={{ today }}";
    let chat = ChatTemplate::new(template);
    let out = chat
        .render(
            &[Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            None,
            None,
            true,
            None,
        )
        .expect("template should render");
    assert!(out.starts_with("Today="));
    assert!(out.len() > "Today=".len());
}

#[test]
fn render_accepts_template_kwargs_and_default_tool_vars() {
    let template = "{% if enable_thinking %}/think{% else %}/no_think{% endif %}|{% if xml_tools %}tools{% else %}no-tools{% endif %}";
    let chat = ChatTemplate::new(template);
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(false));

    let out = chat
        .render(
            &[Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            None,
            None,
            true,
            Some(&kwargs),
        )
        .expect("template should render");
    assert_eq!(out, "/no_think|no-tools");
}

#[test]
fn render_supports_python_string_methods_used_by_smollm3() {
    let template = "{{ '/no_think hello world   '.replace('/no_think', '').rstrip().lstrip(' ') }}|{{ 'aaaab'.lstrip('a') }}";
    let chat = ChatTemplate::new(template);
    let out = chat
        .render(
            &[Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            None,
            None,
            true,
            None,
        )
        .expect("template should render");
    assert_eq!(out, "hello world|b");
}
