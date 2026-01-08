//! Verify per-layer bindings are indexed in the DSL model bindings.

use std::path::Path;

use metallic::{
    MetalError, foundry::{Foundry, model::ModelBuilder}
};
use serial_test::serial;

const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";
const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

#[test]
#[serial]
#[ignore = "requires qwen2.5 model file"]
fn test_layer_weight_bindings_have_indices() -> Result<(), MetalError> {
    let spec_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new()?;
    let model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_gguf(GGUF_PATH)?
        .build(&mut foundry)?;
    let (bindings, _fast_bindings) = model.prepare_bindings(&mut foundry).expect("Bindings failed");

    // First layer bindings should exist.
    assert!(bindings.contains("layer.attn_norm_0"));
    assert!(bindings.contains("layer.attn_q_0"));
    assert!(bindings.contains("layer.ffn_down_0"));

    // Last layer bindings should exist.
    let last = model.architecture().n_layers.saturating_sub(1);
    let last_attn_norm = format!("layer.attn_norm_{}", last);
    let last_attn_q = format!("layer.attn_q_{}", last);
    let last_ffn_down = format!("layer.ffn_down_{}", last);
    assert!(bindings.contains(&last_attn_norm));
    assert!(bindings.contains(&last_attn_q));
    assert!(bindings.contains(&last_ffn_down));

    Ok(())
}
