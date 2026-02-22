#![cfg(test)]

use super::*;

#[derive(serde::Deserialize)]
struct RopeOnly {
    #[serde(default = "default_rope_mode", deserialize_with = "deserialize_rope_mode")]
    rope_mode: DynamicValue<u32>,
    #[serde(default = "default_layer_idx")]
    layer_idx: DynamicValue<u32>,
    #[serde(default = "default_no_rope_layer_step")]
    no_rope_layer_step: DynamicValue<u32>,
}

#[test]
fn rope_mode_defaults_to_normal() {
    let parsed: RopeOnly = serde_json::from_str("{}").unwrap();
    assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NORMAL));
    assert_eq!(parsed.layer_idx, DynamicValue::Literal(0));
    assert_eq!(parsed.no_rope_layer_step, DynamicValue::Literal(0));
}

#[test]
fn rope_mode_accepts_named_literals() {
    let parsed: RopeOnly = serde_json::from_str(r#"{"rope_mode":"neox"}"#).unwrap();
    assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NEOX));
    let parsed: RopeOnly = serde_json::from_str(r#"{"rope_mode":"normal"}"#).unwrap();
    assert_eq!(parsed.rope_mode, DynamicValue::Literal(ROPE_MODE_NORMAL));
}

#[test]
fn no_rope_layer_params_accept_literals_and_dynamic_refs() {
    let parsed: RopeOnly = serde_json::from_str(r#"{"layer_idx":"{i}","no_rope_layer_step":4}"#).unwrap();
    assert_eq!(parsed.layer_idx, DynamicValue::Variable("i".to_string()));
    assert_eq!(parsed.no_rope_layer_step, DynamicValue::Literal(4));
}

#[test]
fn bind_scope_literals_materializes_repeat_index() {
    let params = KvPrepFusedParams {
        layer_idx: DynamicValue::Variable("i".to_string()),
        no_rope_layer_step: DynamicValue::Literal(4),
        ..Default::default()
    };
    let mut bindings = TensorBindings::new();
    bindings.push_scope();
    bindings.set_var("i", "7");
    let bound = params.bind_scope_literals(&bindings);
    assert_eq!(bound.layer_idx, DynamicValue::Literal(7));
    assert_eq!(bound.no_rope_layer_step, DynamicValue::Literal(4));
}
