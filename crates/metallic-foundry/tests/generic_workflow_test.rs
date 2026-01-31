use metallic_foundry::{
    Foundry, model::ModelBuilder, workflow::{WorkflowRunner, WorkflowSpec}
};
use rustc_hash::FxHashMap;
use serial_test::serial;

const MODEL_SPEC_PATH: &str = "../../models/qwen25.json";
const GGUF_PATH: &str = "../../models/qwen2.5-coder-0.5b-instruct-fp16.gguf";

#[test]
#[serial]
#[ignore = "requires qwen2.5 gguf file"]
fn test_generic_workflow_orchestration() -> Result<(), Box<dyn std::error::Error>> {
    let spec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_SPEC_PATH);
    let mut foundry = Foundry::new()?;

    let model = ModelBuilder::new()
        .with_spec_file(&spec_path)?
        .with_gguf(GGUF_PATH)?
        .build(&mut foundry)?;

    // Define a generic workflow in JSON
    let workflow_json = r#"{
        "name": "Generic Orchestration Test",
        "default_model": "main",
        "return_value": "next_text",
        "steps": [
            {
                "op": "tokenize",
                "input": "prompt",
                "output": "tokens"
            },
            {
                "op": "forward",
                "description": "Prefill",
                "inputs": {
                    "input_ids": "tokens"
                },
                "update_globals": {
                    "seq_len": "{tokens.len}",
                    "position_offset": 0
                },
                "apply_derived_globals": true
            },
            {
                "op": "forward",
                "description": "Logits Forward",
                "inputs": {
                    "input_ids": "tokens"
                },
                "outputs": {
                    "logits": "logits_tensor"
                }
            },
            {
                "op": "sample",
                "logits": "logits_tensor",
                "output": "next_token",
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 42
            },
            {
                "op": "detokenize",
                "input": "next_token",
                "output": "next_text"
            },
            {
                "op": "return"
            }
        ]
    }"#;

    let spec: WorkflowSpec = serde_json::from_str(workflow_json)?;

    let mut models = FxHashMap::default();
    models.insert("main".to_string(), &model);

    let mut runner = WorkflowRunner::new(&mut foundry, models);

    let mut inputs = std::collections::HashMap::new();
    inputs.insert("prompt".to_string(), "Hello".to_string());

    let result = runner.run(&spec, inputs, &mut |_, _, _, _| Ok(true))?;

    println!("Workflow result: {:?}", result);
    // Should return text
    match result {
        metallic_foundry::workflow::Value::Text(_) => {}
        _ => panic!("Expected Text result, got {:?}", result),
    }

    Ok(())
}
