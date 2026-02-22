use std::sync::Arc;

use anyhow::Result;
use metallic_foundry::workflow::Value as WorkflowValue;
use rustc_hash::FxHashMap;

pub fn env_bool(name: &str) -> bool {
    let Ok(value) = std::env::var(name) else {
        return false;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
}

pub fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn parse_workflow_kwarg_value(raw: &str) -> WorkflowValue {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("true") {
        return WorkflowValue::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return WorkflowValue::Bool(false);
    }
    if let Ok(v) = trimmed.parse::<u32>() {
        return WorkflowValue::U32(v);
    }
    if let Ok(v) = trimmed.parse::<usize>() {
        return WorkflowValue::Usize(v);
    }
    if let Ok(v) = trimmed.parse::<f32>() {
        return WorkflowValue::F32(v);
    }
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        return WorkflowValue::from_json(json);
    }
    WorkflowValue::Text(Arc::<str>::from(trimmed.to_string()))
}

pub fn build_workflow_cli_inputs(kwargs: &[(String, String)], thinking_override: Option<bool>) -> Result<FxHashMap<String, WorkflowValue>> {
    let mut out: FxHashMap<String, WorkflowValue> = FxHashMap::default();
    for (key, value) in kwargs {
        out.insert(key.clone(), parse_workflow_kwarg_value(value));
    }
    if let Some(enabled) = thinking_override {
        out.insert("enable_thinking".to_string(), WorkflowValue::U32(u32::from(enabled)));
    }
    Ok(out)
}

pub fn summarize_workflow_value(value: &WorkflowValue) -> String {
    match value {
        WorkflowValue::Bool(v) => v.to_string(),
        WorkflowValue::U32(v) => v.to_string(),
        WorkflowValue::Usize(v) => v.to_string(),
        WorkflowValue::F32(v) => v.to_string(),
        WorkflowValue::Text(v) => {
            let text = v.as_ref();
            if text.len() > 64 {
                format!("{}…", &text[..64])
            } else {
                text.to_string()
            }
        }
        WorkflowValue::Array(items) => format!("<array:{}>", items.len()),
        WorkflowValue::Map(map) => format!("<map:{}>", map.len()),
        WorkflowValue::TokensU32(tokens) => format!("<tokens:{}>", tokens.len()),
        WorkflowValue::Tensor(_) => "<tensor>".to_string(),
        WorkflowValue::ChannelU32(_) => "<channel_u32>".to_string(),
        WorkflowValue::CommandBuffer(_) => "<command_buffer>".to_string(),
    }
}

fn workflow_value_to_json(value: &WorkflowValue) -> Result<serde_json::Value> {
    Ok(match value {
        WorkflowValue::U32(v) => serde_json::Value::from(*v),
        WorkflowValue::Usize(v) => serde_json::Value::from(*v),
        WorkflowValue::F32(v) => serde_json::Value::from(*v),
        WorkflowValue::Bool(v) => serde_json::Value::from(*v),
        WorkflowValue::Text(v) => serde_json::Value::String(v.to_string()),
        WorkflowValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(workflow_value_to_json(item)?);
            }
            serde_json::Value::Array(out)
        }
        WorkflowValue::Map(map) => {
            let mut out = serde_json::Map::new();
            for (key, item) in map {
                out.insert(key.clone(), workflow_value_to_json(item)?);
            }
            serde_json::Value::Object(out)
        }
        WorkflowValue::TokensU32(tokens) => serde_json::Value::Array(tokens.iter().map(|v| serde_json::Value::from(*v)).collect()),
        WorkflowValue::Tensor(_) => return Err(anyhow::anyhow!("Cannot convert tensor workflow kwarg to template JSON")),
        WorkflowValue::ChannelU32(_) => return Err(anyhow::anyhow!("Cannot convert channel workflow kwarg to template JSON")),
        WorkflowValue::CommandBuffer(_) => {
            return Err(anyhow::anyhow!("Cannot convert command_buffer workflow kwarg to template JSON"));
        }
    })
}

pub fn workflow_template_kwargs(
    workflow_cli_inputs: &FxHashMap<String, WorkflowValue>,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>> {
    if workflow_cli_inputs.is_empty() {
        return Ok(None);
    }
    let mut out = serde_json::Map::new();
    for (key, value) in workflow_cli_inputs {
        out.insert(key.clone(), workflow_value_to_json(value)?);
    }
    Ok(Some(out))
}

pub fn apply_workflow_cli_inputs(inputs: &mut FxHashMap<String, WorkflowValue>, workflow_cli_inputs: &FxHashMap<String, WorkflowValue>) {
    for (key, value) in workflow_cli_inputs {
        inputs.insert(key.clone(), value.clone());
    }
}
