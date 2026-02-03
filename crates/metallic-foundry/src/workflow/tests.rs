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

    fn mutate_step_json(step: &mut serde_json::Value) {
        let serde_json::Value::Object(map) = step else { return };
        let op = map.get("op").and_then(|v| v.as_str()).unwrap_or("");

        if op == "graph_forward" {
            map.insert("logits_binding".to_string(), serde_json::Value::String("not_logits".to_string()));
            return;
        }

        match op {
            "if" => {
                for key in ["then", "else"] {
                    let Some(serde_json::Value::Array(arr)) = map.get_mut(key) else {
                        continue;
                    };
                    for child in arr {
                        mutate_step_json(child);
                    }
                }
            }
            "while" => {
                for key in ["body", "phases"] {
                    let Some(serde_json::Value::Array(arr)) = map.get_mut(key) else {
                        continue;
                    };
                    for child in arr {
                        mutate_step_json(child);
                    }
                }
            }
            _ => {}
        }
    }

    fn mutate_steps(steps: &mut [super::WorkflowStepSpec]) {
        for step in steps {
            if step.op == "graph_forward" {
                if let serde_json::Value::Object(map) = &mut step.params {
                    map.insert("logits_binding".to_string(), serde_json::Value::String("not_logits".to_string()));
                }
                continue;
            }

            let serde_json::Value::Object(map) = &mut step.params else {
                continue;
            };
            match step.op.as_str() {
                "if" => {
                    for key in ["then", "else"] {
                        let Some(serde_json::Value::Array(arr)) = map.get_mut(key) else {
                            continue;
                        };
                        for child in arr {
                            mutate_step_json(child);
                        }
                    }
                }
                "while" => {
                    for key in ["body", "phases"] {
                        let Some(serde_json::Value::Array(arr)) = map.get_mut(key) else {
                            continue;
                        };
                        for child in arr {
                            mutate_step_json(child);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    mutate_steps(&mut spec.steps);

    let msg = match CompiledWorkflow::compile(&spec) {
        Ok(_) => panic!("expected compile to fail"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("graph_forward.logits_binding"), "got: {msg}");
}
