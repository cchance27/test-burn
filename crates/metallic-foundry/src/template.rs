use std::sync::{Arc, OnceLock};

use chrono::Local;
use minijinja::{Error, ErrorKind, Value};
use regex::Regex;
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

fn generation_tag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").expect("valid generation tag regex"))
}

fn normalize_template(template: &str) -> Arc<str> {
    let normalized = generation_tag_regex().replace_all(template, "");
    Arc::<str>::from(normalized.into_owned())
}

impl ChatTemplate {
    pub fn new(template: &str) -> Self {
        let mut env = minijinja::Environment::new();
        env.set_keep_trailing_newline(true);
        // HF chat templates rely on this function.
        env.add_function("strftime_now", |fmt: String| -> String { Local::now().format(&fmt).to_string() });
        env.set_unknown_method_callback(|_state, value, method, args| {
            let text = value.as_str().ok_or_else(|| Error::from(ErrorKind::UnknownMethod))?;
            match method {
                "replace" => {
                    if !(2..=3).contains(&args.len()) {
                        return Err(Error::new(ErrorKind::InvalidOperation, "string.replace expects 2 or 3 arguments"));
                    }

                    let old = args[0]
                        .as_str()
                        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string.replace old must be a string"))?;
                    let new = args[1]
                        .as_str()
                        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string.replace new must be a string"))?;
                    let replaced = if let Some(count) = args.get(2) {
                        let count = count
                            .as_i64()
                            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string.replace count must be an integer"))?;
                        if count < 0 {
                            text.replace(old, new)
                        } else {
                            text.replacen(old, new, count as usize)
                        }
                    } else {
                        text.replace(old, new)
                    };
                    Ok(Value::from(replaced))
                }
                "lstrip" => {
                    if args.len() > 1 {
                        return Err(Error::new(ErrorKind::InvalidOperation, "string.lstrip expects 0 or 1 argument"));
                    }
                    let stripped = if let Some(chars) = args.first() {
                        let chars = chars
                            .as_str()
                            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string.lstrip chars must be a string"))?;
                        text.trim_start_matches(|c| chars.contains(c)).to_string()
                    } else {
                        text.trim_start().to_string()
                    };
                    Ok(Value::from(stripped))
                }
                "rstrip" => {
                    if args.len() > 1 {
                        return Err(Error::new(ErrorKind::InvalidOperation, "string.rstrip expects 0 or 1 argument"));
                    }
                    let stripped = if let Some(chars) = args.first() {
                        let chars = chars
                            .as_str()
                            .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "string.rstrip chars must be a string"))?;
                        text.trim_end_matches(|c| chars.contains(c)).to_string()
                    } else {
                        text.trim_end().to_string()
                    };
                    Ok(Value::from(stripped))
                }
                _ => Err(Error::from(ErrorKind::UnknownMethod)),
            }
        });

        let template_str = normalize_template(template);

        Self { env, template_str }
    }

    #[inline]
    pub fn raw(&self) -> &str {
        &self.template_str
    }

    pub fn render(
        &self,
        messages: &[Message],
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, MetalError> {
        if self.template_str.contains("bos_token") && bos_token.is_none() {
            return Err(MetalError::InvalidOperation(
                "Chat template requires 'bos_token' but tokenizer metadata/vocab did not provide it".to_string(),
            ));
        }
        if self.template_str.contains("eos_token") && eos_token.is_none() {
            return Err(MetalError::InvalidOperation(
                "Chat template requires 'eos_token' but tokenizer metadata/vocab did not provide it".to_string(),
            ));
        }
        let bos_token = bos_token.unwrap_or("");
        let eos_token = eos_token.unwrap_or("");
        let mut ctx = serde_json::Map::new();
        ctx.insert(
            "messages".to_string(),
            serde_json::to_value(messages)
                .map_err(|e| MetalError::InvalidOperation(format!("Template messages serialization failed: {e}")))?,
        );
        ctx.insert("bos_token".to_string(), serde_json::Value::String(bos_token.to_string()));
        ctx.insert("eos_token".to_string(), serde_json::Value::String(eos_token.to_string()));
        ctx.insert("add_generation_prompt".to_string(), serde_json::Value::Bool(add_generation_prompt));

        if let Some(kwargs) = template_kwargs {
            for (k, v) in kwargs {
                ctx.insert(k.clone(), v.clone());
            }
        }

        // Some templates branch on `tools is not none`; default to null (not empty array)
        // so we don't accidentally trigger tool-calling JSON mode when no tools were provided.
        // Keep these names defined for templates that check them conditionally.
        for key in ["xml_tools", "python_tools", "tools"] {
            if !ctx.contains_key(key) {
                ctx.insert(key.to_string(), serde_json::Value::Null);
            }
        }

        self.env
            .render_str(&self.template_str, serde_json::Value::Object(ctx))
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

#[cfg(test)]
mod tests {
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
}
