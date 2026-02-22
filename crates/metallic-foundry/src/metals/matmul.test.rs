#![cfg(test)]

use super::*;

#[test]
fn matmul_step_defaults_round_trip() {
    let json = r#"{
        "a": "a",
        "b": "b",
        "output": "output",
        "m": 1,
        "n": 1,
        "k": 1,
        "transpose_a": false,
        "transpose_b": false,
        "weights_per_block": 32
    }"#;

    let step: MatMulStep = serde_json::from_str(json).unwrap();
    assert_eq!(step.alpha, 1.0);
    assert_eq!(step.beta, 0.0);
}

#[test]
fn matmul_step_respects_explicit_alpha_beta() {
    let json = r#"{
        "a": "a",
        "b": "b",
        "output": "output",
        "m": 1,
        "n": 1,
        "k": 1,
        "transpose_a": false,
        "transpose_b": false,
        "alpha": 0.25,
        "beta": 0.75
    }"#;

    let step: MatMulStep = serde_json::from_str(json).unwrap();
    assert_eq!(step.alpha, 0.25);
    assert_eq!(step.beta, 0.75);
}
