#![cfg(test)]

use super::*;

#[test]
fn int_expr_literal_eval() {
    let expr: IntExpr = serde_json::from_str("128").unwrap();
    let bindings = TensorBindings::new();
    assert_eq!(expr.eval(&bindings), 128);
}

#[test]
fn int_expr_var_and_ops_eval() {
    let expr: IntExpr = serde_json::from_str(r#""m * (d_model / n_heads)""#).unwrap();
    let mut bindings = TensorBindings::new();
    bindings.set_int_global("m", 32);
    bindings.set_int_global("d_model", 896);
    bindings.set_int_global("n_heads", 14);
    assert_eq!(expr.eval(&bindings), 32 * (896 / 14));
}

#[test]
fn int_expr_braced_vars_eval() {
    let expr: IntExpr = serde_json::from_str(r#""{position_offset} + {seq_len}""#).unwrap();
    let mut bindings = TensorBindings::new();
    bindings.set_int_global("position_offset", 10);
    bindings.set_int_global("seq_len", 7);
    assert_eq!(expr.eval(&bindings), 17);
}

#[test]
#[should_panic]
fn int_expr_missing_var_panics() {
    let expr: IntExpr = serde_json::from_str(r#""m * d_model""#).unwrap();
    let bindings = TensorBindings::new();
    let _ = expr.eval(&bindings);
}

#[test]
#[should_panic]
fn int_expr_overflow_panics() {
    let expr: IntExpr = serde_json::from_str(r#""18446744073709551615 * 2""#).unwrap();
    let bindings = TensorBindings::new();
    let _ = expr.eval(&bindings);
}
