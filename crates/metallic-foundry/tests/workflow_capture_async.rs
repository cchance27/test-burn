use std::sync::Arc;

use metallic_foundry::{
    Foundry, workflow::{Value, WorkflowRunner, WorkflowSpec}
};
use rustc_hash::FxHashMap;

#[test]
fn workflow_capture_end_wait_false_then_wait_executes_kernels() {
    let mut foundry = Foundry::new().expect("foundry init");
    let models: FxHashMap<String, Arc<metallic_foundry::model::CompiledModel>> = FxHashMap::default();
    let mut runner = WorkflowRunner::new(&mut foundry, models);

    let workflow_json = r#"
{
  "name": "CaptureAsync",
  "inputs": [],
  "steps": [
    {"op":"stream_init","output":"token_stream","capacity":64},
    {"op":"capture_begin"},
    {"op":"stream_write_u32","channel":"token_stream","input":"next_token"},
    {"op":"capture_end","wait":false,"output":"cmd"},
    {"op":"capture_wait","input":"cmd"},
    {"op":"return","output":"token_stream"}
  ]
}
"#;
    let spec: WorkflowSpec = serde_json::from_str(workflow_json).expect("parse workflow json");
    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    inputs.insert("next_token".to_string(), Value::U32(123));

    let out = runner
        .run_streaming(&spec, inputs, |_tok, _prefill, _setup, _iter| Ok(true))
        .expect("run");

    let chan = out
        .get("token_stream")
        .and_then(|v| v.as_channel_u32())
        .cloned()
        .expect("token_stream present");
    let mut reader = metallic_foundry::workflow::ChannelU32Reader::new(chan);
    let mut drained = Vec::new();
    reader.drain_into(&mut drained).expect("drain");
    assert_eq!(drained, vec![123]);
}
