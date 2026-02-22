#![cfg(test)]

use super::*;

#[test]
fn test_dynamic_value_literal_u32() {
    let json = "42";
    let val: DynamicValue<u32> = serde_json::from_str(json).unwrap();
    assert_eq!(val, DynamicValue::Literal(42));
}

#[test]
fn test_dynamic_value_variable() {
    let json = r#""{position_offset}""#;
    let val: DynamicValue<u32> = serde_json::from_str(json).unwrap();
    assert_eq!(val, DynamicValue::Variable("position_offset".to_string()));
}

#[test]
fn test_dynamic_value_resolve_literal() {
    let val = DynamicValue::Literal(42u32);
    let bindings = TensorBindings::new();
    assert_eq!(val.resolve(&bindings), 42);
}

#[test]
fn test_dynamic_value_resolve_variable() {
    let val = DynamicValue::<u32>::Variable("test_var".to_string());
    let mut bindings = TensorBindings::new();
    bindings.set_global("test_var", "123".to_string());
    assert_eq!(val.resolve(&bindings), 123);
}

#[test]
#[should_panic]
fn test_dynamic_value_resolve_missing_variable_panics() {
    let val = DynamicValue::<u32>::Variable("missing".to_string());
    let bindings = TensorBindings::new();
    let _ = val.resolve(&bindings);
}

#[test]
fn test_dynamic_value_serialize_literal() {
    let val = DynamicValue::Literal(42u32);
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "42");
}

#[test]
fn test_dynamic_value_serialize_variable() {
    let val = DynamicValue::<u32>::Variable("pos".to_string());
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, r#""{pos}""#);
}
