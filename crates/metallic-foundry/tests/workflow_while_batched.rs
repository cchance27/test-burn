use std::sync::Arc;

use metallic_foundry::{
    Foundry, workflow::{Value, WorkflowRunner, WorkflowSpec, register_op}
};
use rustc_hash::FxHashMap;

#[test]
fn while_batched_trims_after_eos_at_batch_boundary() {
    // Register a test-only op that produces a deterministic token sequence by writing a u32 into a 1-token buffer.
    // This avoids relying on model execution while still exercising WhileBatchedOp's batching + EOS trimming logic.
    register_op("test_set_token_tensor", |_| {
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
                // Sequence: 1, EOS(2), 3, 4...
                self.i = self.i.wrapping_add(1);
                let token = if self.i == 2 { 2 } else { self.i };

                let buf = ctx
                    .foundry
                    .device
                    .new_buffer(4, metallic_foundry::types::MetalResourceOptions::StorageModeShared)
                    .expect("alloc token buf");
                buf.copy_from_slice(&[token]);
                let arg = metallic_foundry::types::TensorArg::from_buffer(buf, metallic_foundry::tensor::Dtype::U32, vec![1], vec![1]);
                ctx.values.insert("next_token".to_string(), Value::Tensor(arg));
                Ok(metallic_foundry::workflow::ops::WorkflowOpOutcome::Continue)
            }
        }
        Ok(Box::new(Op { i: 0 }))
    });

    // Ensure EOS stop is enabled for this test (default behavior).
    unsafe {
        std::env::remove_var("METALLIC_IGNORE_EOS_STOP");
        std::env::set_var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE", "4");
    }

    let mut foundry = Foundry::new().expect("foundry init");
    let models: FxHashMap<String, Arc<metallic_foundry::model::CompiledModel>> = FxHashMap::default();
    let mut runner = WorkflowRunner::new(models);

    // Workflow: run while_batched for up to 8 iterations, producing tokens via test op.
    let workflow_json = r#"
{
  "name": "WhileBatchedTrimEOS",
  "inputs": [
    {"name":"max_tokens","type":"u32","default":8},
    {"name":"eos_token","type":"u32","default":2}
  ],
  "steps": [
    {
      "op": "while_batched",
      "condition": "max_tokens",
      "max_iterations": "{max_tokens}",
      "unsafe_allow_overshoot": true,
      "token_var": "next_token",
      "output_tokens": "generated_tokens",
      "eos_token": "{eos_token}",
      "body": [
        {"op":"test_set_token_tensor"}
      ]
    },
    {"op":"return","output":"generated_tokens"}
  ]
}
"#;

    let spec: WorkflowSpec = serde_json::from_str(workflow_json).expect("parse workflow json");
    let mut inputs: FxHashMap<String, Value> = FxHashMap::default();
    inputs.insert("max_tokens".to_string(), Value::U32(8));
    inputs.insert("eos_token".to_string(), Value::U32(2));

    let mut seen: Vec<u32> = Vec::new();
    let out = runner
        .run_streaming(&mut foundry, &spec, inputs, |tok, _prefill, _setup, _iter| {
            seen.push(tok);
            Ok(true)
        })
        .expect("run");

    let tokens = out
        .get("generated_tokens")
        .and_then(|v| v.as_tokens_u32())
        .expect("generated_tokens present");

    // Expect: 1 then stop BEFORE emitting EOS(2), trimming any overshoot tokens.
    assert_eq!(tokens, &[1]);
    assert_eq!(seen, vec![1]);
}
