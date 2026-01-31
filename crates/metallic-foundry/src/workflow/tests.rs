use super::{WorkflowSpec, compiler::CompiledWorkflow};

#[test]
fn text_generation_workflow_parses_and_compiles() {
    let json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workflows/text_generation.json"));
    let spec: WorkflowSpec = serde_json::from_str(json).expect("workflow JSON must parse");

    CompiledWorkflow::compile(&spec).expect("workflow must compile");
}

#[test]
fn workflow_compile_fails_on_mismatched_logits_binding() {
    let json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workflows/text_generation.json"));
    let mut spec: WorkflowSpec = serde_json::from_str(json).expect("workflow JSON must parse");

    // Mutate graph_forward logits_binding via params JSON.
    for step in &mut spec.steps {
        if step.op == "graph_forward" {
            if let serde_json::Value::Object(map) = &mut step.params {
                map.insert("logits_binding".to_string(), serde_json::Value::String("not_logits".to_string()));
            }
        }
    }

    let msg = match CompiledWorkflow::compile(&spec) {
        Ok(_) => panic!("expected compile to fail"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("graph_forward.logits_binding"), "got: {msg}");
}
