use std::sync::Arc;

use metallic_env::{EnvVarGuard, FoundryEnvVar};
use metallic_foundry::{
    Foundry, workflow::{Value, WorkflowRunner, WorkflowSpec, register_op}
};
use rustc_hash::FxHashMap;

#[test]
fn workflow_stream_u32_emits_tokens_via_channel() {
    // Test-only op: writes a deterministic scalar token (u32) into "next_token".
    register_op("test_set_token_u32", |_| {
        struct Op {
            i: u32,
        }
        impl metallic_foundry::workflow::ops::WorkflowOp for Op {
            fn execute(
                &mut self,
                ctx: &mut metallic_foundry::workflow::WorkflowExecutionContext<'_>,
                _on_token: &mut dyn FnMut(
                    u32,
                    std::time::Duration,
                    std::time::Duration,
                    Option<std::time::Duration>,
                ) -> Result<bool, metallic_foundry::MetalError>,
            ) -> Result<metallic_foundry::workflow::ops::WorkflowOpOutcome, metallic_foundry::MetalError> {
                self.i = self.i.wrapping_add(1);
                ctx.values.insert("next_token".to_string(), Value::U32(self.i));
                Ok(metallic_foundry::workflow::ops::WorkflowOpOutcome::Continue)
            }
        }
        Ok(Box::new(Op { i: 0 }))
    });

    // Ensure batching doesn't trip guardrails in this test.
    let _ignore_eos_guard = EnvVarGuard::set(FoundryEnvVar::IgnoreEosStop, "1");
    let _decode_batch_guard = EnvVarGuard::set(FoundryEnvVar::FoundryDecodeBatchSize, "8");

    let mut foundry = Foundry::new().expect("foundry init");
    let models: FxHashMap<String, Arc<metallic_foundry::model::CompiledModel>> = FxHashMap::default();
    let mut runner = WorkflowRunner::new(models);

    let workflow_json = r#"
{
  "name": "StreamU32",
  "inputs": [{"name":"max_tokens","type":"u32","default":16},{"name":"stream_capacity","type":"u32","default":64}],
  "steps": [
    {"op":"stream_init","output":"token_stream","capacity":"{stream_capacity}"},
    {
      "op":"while_batched",
      "condition":"max_tokens",
      "max_iterations":"{max_tokens}",
      "unsafe_allow_overshoot": true,
      "token_var":"next_token",
      "stream_channel":"token_stream",
      "output_tokens":"generated_tokens",
      "body":[
        {"op":"test_set_token_u32"},
        {"op":"stream_write_u32","channel":"token_stream","input":"next_token"}
      ]
    },
    {"op":"return","output":"generated_tokens"}
  ]
}
"#;
    let spec: WorkflowSpec = serde_json::from_str(workflow_json).expect("parse workflow json");
    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    inputs.insert("max_tokens".to_string(), Value::U32(16));
    inputs.insert("stream_capacity".to_string(), Value::U32(64));

    let mut streamed: Vec<u32> = Vec::new();
    let out = runner
        .run_streaming(&mut foundry, &spec, inputs, |tok, _prefill, _setup, _iter| {
            streamed.push(tok);
            Ok(true)
        })
        .expect("run");

    let chan = out
        .get("token_stream")
        .and_then(|v| v.as_channel_u32())
        .cloned()
        .expect("token_stream present");
    let mut reader = metallic_foundry::workflow::ChannelU32Reader::new(chan);
    let mut streamed_from_channel: Vec<u32> = Vec::new();
    reader.drain_into(&mut streamed_from_channel).expect("drain channel");

    let generated = out
        .get("generated_tokens")
        .and_then(|v| v.as_tokens_u32())
        .expect("generated_tokens present");

    assert_eq!(generated, (1u32..=16u32).collect::<Vec<_>>());
    assert_eq!(streamed, (1u32..=16u32).collect::<Vec<_>>());
    assert_eq!(streamed_from_channel, (1u32..=16u32).collect::<Vec<_>>());
}
