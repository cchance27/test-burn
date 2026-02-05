use std::sync::Arc;

use metallic_loader::ModelLoader;
use rustc_hash::FxHashMap;
use tracing;

use super::{
    Value, compiler::CompiledWorkflow, ops::WorkflowOpOutcome, spec::{Param, WorkflowSpec}
};
use crate::{
    Foundry, error::MetalError, model::{CompiledModel, ModelBuilder}
};

pub struct WorkflowRunnerConfig {
    pub workflow: WorkflowSpec,
}

pub struct WorkflowExecutionContext<'a> {
    pub workflow: &'a WorkflowSpec,
    pub foundry: &'a mut Foundry,
    pub models: &'a FxHashMap<String, Arc<CompiledModel>>,
    pub values: FxHashMap<String, Value>,
    pub return_key: Option<String>,
}

impl<'a> WorkflowExecutionContext<'a> {
    pub fn resolve_model(&self, model_id: Option<&str>) -> Result<Arc<CompiledModel>, MetalError> {
        let id = if let Some(id) = model_id {
            id
        } else if let Some(default) = self.workflow.default_model.as_deref() {
            default
        } else if self.models.len() == 1 {
            self.models.keys().next().map(|s| s.as_str()).unwrap()
        } else {
            return Err(MetalError::InvalidOperation(
                "Workflow step missing model_id and workflow.default_model is not set".into(),
            ));
        };

        self.models
            .get(id)
            .cloned()
            .ok_or_else(|| MetalError::InvalidOperation(format!("Workflow references unknown model_id '{id}'")))
    }

    pub fn resolve_param_u32(&self, p: &Param<u32>) -> Result<u32, MetalError> {
        match p {
            Param::Literal(v) => Ok(*v),
            Param::Input(name) => {
                if let Some((var_name, "len")) = name.split_once('.') {
                    let val = self
                        .values
                        .get(var_name)
                        .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow variable '{}' for .len", var_name)))?;
                    match val {
                        Value::TokensU32(t) => Ok(t.len() as u32),
                        Value::Tensor(t) => Ok(t.dims().iter().product::<usize>() as u32),
                        _ => Err(MetalError::InvalidOperation(format!(
                            "Variable '{}' of type {} does not have .len",
                            var_name,
                            val.type_name()
                        ))),
                    }
                } else {
                    self.values
                        .get(name)
                        .and_then(|v| match v {
                            Value::U32(v) => Some(*v),
                            Value::Usize(v) => (*v).try_into().ok(),
                            _ => None,
                        })
                        .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{}' (u32)", name)))
                }
            }
        }
    }

    pub fn resolve_param_usize(&self, p: &Param<usize>) -> Result<usize, MetalError> {
        match p {
            Param::Literal(v) => Ok(*v),
            Param::Input(name) => {
                if let Some((var_name, "len")) = name.split_once('.') {
                    let val = self
                        .values
                        .get(var_name)
                        .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow variable '{}' for .len", var_name)))?;
                    match val {
                        Value::TokensU32(t) => Ok(t.len()),
                        Value::Tensor(t) => Ok(t.dims().iter().product::<usize>()),
                        _ => Err(MetalError::InvalidOperation(format!(
                            "Variable '{}' of type {} does not have .len",
                            var_name,
                            val.type_name()
                        ))),
                    }
                } else {
                    self.values
                        .get(name)
                        .and_then(|v| v.as_usize())
                        .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{}' (usize)", name)))
                }
            }
        }
    }

    pub fn resolve_param_f32(&self, p: &Param<f32>) -> Result<f32, MetalError> {
        match p {
            Param::Literal(v) => Ok(*v),
            Param::Input(name) => self
                .values
                .get(name)
                .and_then(|v| match v {
                    Value::F32(v) => Some(*v),
                    _ => None,
                })
                .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{}' (f32)", name))),
        }
    }

    pub fn read_usize(&self, name: &str) -> Result<usize, MetalError> {
        if let Some((var_name, "len")) = name.split_once('.') {
            let val = self
                .values
                .get(var_name)
                .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow variable '{}' for .len", var_name)))?;
            match val {
                Value::TokensU32(t) => Ok(t.len()),
                Value::Tensor(t) => Ok(t.dims().iter().product::<usize>()),
                _ => Err(MetalError::InvalidOperation(format!(
                    "Variable '{}' of type {} does not have .len",
                    var_name,
                    val.type_name()
                ))),
            }
        } else {
            self.values
                .get(name)
                .and_then(|v| match v {
                    Value::Usize(v) => Some(*v),
                    Value::U32(v) => Some(*v as usize),
                    _ => None,
                })
                .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{}' (usize/u32)", name)))
        }
    }
}

pub struct WorkflowRunner {
    models: FxHashMap<String, Arc<CompiledModel>>,
    compiled: Option<(u64, CompiledWorkflow)>,
}

fn value_u32_like(v: &Value) -> Option<u32> {
    match v {
        Value::U32(n) => Some(*n),
        Value::Usize(n) => (*n).try_into().ok(),
        _ => None,
    }
}

fn workflow_input_default_u32(workflow: &WorkflowSpec, name: &str) -> Option<u32> {
    workflow
        .inputs
        .iter()
        .find(|i| i.name == name)
        .and_then(|i| i.default.as_ref())
        .and_then(|d| d.as_u64())
        .and_then(|n| u32::try_from(n).ok())
}

fn should_override_with_inferred_eos(
    eos_was_explicitly_supplied: bool,
    current_eos: Option<u32>,
    workflow_default_eos: Option<u32>,
) -> bool {
    if !eos_was_explicitly_supplied {
        return true;
    }
    matches!((current_eos, workflow_default_eos), (Some(current), Some(default)) if current == default)
}

impl WorkflowRunner {
    pub fn new(models: FxHashMap<String, Arc<CompiledModel>>) -> Self {
        Self { models, compiled: None }
    }

    pub fn load_resources(&mut self, foundry: &mut Foundry, spec: &WorkflowSpec) -> Result<(), MetalError> {
        if let Some(resources) = &spec.resources {
            for model_spec in &resources.models {
                if self.models.contains_key(&model_spec.id) {
                    continue; // Already loaded or injected
                }

                let model_loaded = ModelLoader::from_file(&model_spec.gguf_path)
                    .map_err(|e| MetalError::OperationFailed(format!("Model load failed: {}", e)))?;
                let model = ModelBuilder::new()
                    .with_spec_file(std::path::PathBuf::from(&model_spec.spec_path))?
                    .with_model(model_loaded)
                    .build(foundry)?;

                self.models.insert(model_spec.id.clone(), Arc::new(model));
            }
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        foundry: &mut Foundry,
        spec: &WorkflowSpec,
        mut inputs_raw: std::collections::HashMap<String, String>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<Value, MetalError> {
        let mut inputs = FxHashMap::default();
        for (k, v) in inputs_raw.drain() {
            inputs.insert(k, Value::Text(v.into()));
        }

        let values = self.run_streaming(foundry, spec, inputs, on_token)?;

        if let Some(return_key) = &spec.return_value {
            values
                .get(return_key)
                .cloned()
                .ok_or_else(|| MetalError::InvalidOperation(format!("Workflow return_value key '{}' not found", return_key)))
        } else {
            Ok(Value::U32(0))
        }
    }

    /// Resets the internal state of the compiled workflow's operations (e.g. chat history counters).
    pub fn reset(&mut self) {
        if let Some((_, compiled)) = &mut self.compiled {
            compiled.reset();
        }
    }

    pub fn run_streaming(
        &mut self,
        foundry: &mut Foundry,
        workflow: &WorkflowSpec,
        mut inputs: FxHashMap<String, Value>,
        mut on_token: impl FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<FxHashMap<String, Value>, MetalError> {
        fn hash_json_value(h: &mut rustc_hash::FxHasher, v: &serde_json::Value) {
            use std::hash::Hash;
            match v {
                serde_json::Value::Null => 0u8.hash(h),
                serde_json::Value::Bool(b) => {
                    1u8.hash(h);
                    b.hash(h);
                }
                serde_json::Value::Number(n) => {
                    2u8.hash(h);
                    // Stable representation across int/float types.
                    n.to_string().hash(h);
                }
                serde_json::Value::String(s) => {
                    3u8.hash(h);
                    s.hash(h);
                }
                serde_json::Value::Array(arr) => {
                    4u8.hash(h);
                    arr.len().hash(h);
                    for item in arr {
                        hash_json_value(h, item);
                    }
                }
                serde_json::Value::Object(map) => {
                    5u8.hash(h);
                    map.len().hash(h);
                    // Deterministic order: JSON object key order is not guaranteed.
                    let mut keys: Vec<&str> = map.keys().map(|s| s.as_str()).collect();
                    keys.sort_unstable();
                    for k in keys {
                        k.hash(h);
                        if let Some(val) = map.get(k) {
                            hash_json_value(h, val);
                        }
                    }
                }
            }
        }

        fn workflow_fingerprint(workflow: &WorkflowSpec) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            workflow.name.hash(&mut h);
            workflow.default_model.hash(&mut h);
            workflow.return_value.hash(&mut h);

            workflow.inputs.len().hash(&mut h);
            for inp in &workflow.inputs {
                inp.name.hash(&mut h);
                inp.ty.hash(&mut h);
                if let Some(d) = &inp.default {
                    hash_json_value(&mut h, d);
                } else {
                    0u8.hash(&mut h);
                }
            }

            workflow.steps.len().hash(&mut h);
            for step in &workflow.steps {
                step.op.hash(&mut h);
                hash_json_value(&mut h, &step.params);
            }

            h.finish()
        }

        // Ensure any resources defined in the workflow are loaded.
        self.load_resources(foundry, workflow)?;

        let eos_was_explicitly_supplied = inputs.contains_key("eos_token");

        // Apply defaults from workflow inputs.
        for inp in &workflow.inputs {
            if inputs.contains_key(&inp.name) {
                continue;
            }
            let Some(default) = &inp.default else { continue };
            match inp.ty.as_str() {
                "u32" => {
                    if let Some(v) = default.as_u64() {
                        inputs.insert(inp.name.clone(), Value::U32(v as u32));
                    }
                }
                "usize" => {
                    if let Some(v) = default.as_u64() {
                        inputs.insert(inp.name.clone(), Value::Usize(v as usize));
                    }
                }
                "f32" => {
                    if let Some(v) = default.as_f64() {
                        inputs.insert(inp.name.clone(), Value::F32(v as f32));
                    }
                }
                _ => {}
            }
        }

        // Convenience: infer `eos_token` from the default model's tokenizer.
        //
        // Rationale:
        // - Workflows are model-agnostic, but token IDs (EOS/BOS/etc.) are tokenizer-defined.
        // - Callers often persist workflow defaults as "inputs", which should not pin an incorrect
        //   model-specific EOS value (e.g. 151645 for non-Qwen models).
        // - If multiple models are loaded and `workflow.default_model` is missing, we cannot infer safely.
        if workflow.inputs.iter().any(|i| i.name == "eos_token") {
            let model_id = if let Some(id) = workflow.default_model.as_deref() {
                Some(id)
            } else if self.models.len() == 1 {
                self.models.keys().next().map(|s| s.as_str())
            } else {
                None
            };

            if let Some(model_id) = model_id
                && let Some(model) = self.models.get(model_id)
                && let Ok(tok) = model.tokenizer()
                && let Some(eos) = tok.special_tokens().eos_token_id
            {
                let workflow_default_eos = workflow_input_default_u32(workflow, "eos_token");
                let current_eos = inputs.get("eos_token").and_then(value_u32_like);
                if should_override_with_inferred_eos(eos_was_explicitly_supplied, current_eos, workflow_default_eos) {
                    if let Some(existing) = current_eos
                        && existing != eos
                    {
                        tracing::debug!(
                            "Overriding eos_token {} -> {} using tokenizer metadata for model '{}'",
                            existing,
                            eos,
                            model_id
                        );
                    }
                    inputs.insert("eos_token".to_string(), Value::U32(eos));
                }
            }
        }

        let fp = workflow_fingerprint(workflow);
        let needs_compile = self.compiled.as_ref().map(|(old, _)| *old != fp).unwrap_or(true);
        if needs_compile {
            self.compiled = Some((fp, CompiledWorkflow::compile(workflow)?));
        }
        let compiled = self.compiled.as_mut().expect("compiled set").1.ops.as_mut_slice();

        let mut ctx = WorkflowExecutionContext {
            workflow,
            foundry,
            models: &self.models,
            values: inputs,
            return_key: None,
        };

        let debug_ops = std::env::var("METALLIC_DEBUG_WORKFLOW_OPS").is_ok();
        for cop in compiled.iter_mut() {
            if debug_ops {
                tracing::info!(target: "metallic_foundry::workflow::runner", "Workflow begin_run op={}", cop.name);
            }
            cop.op.begin_run(&mut ctx)?;
        }

        for cop in compiled.iter_mut() {
            if debug_ops {
                tracing::info!(target: "metallic_foundry::workflow::runner", "Workflow execute op={}", cop.name);
            }
            match cop.op.execute(&mut ctx, &mut on_token)? {
                WorkflowOpOutcome::Continue => {}
                WorkflowOpOutcome::Return => break,
                WorkflowOpOutcome::Break => return Err(MetalError::InvalidOperation("Unexpected 'break' outside of loop".into())),
                WorkflowOpOutcome::LoopContinue => {
                    return Err(MetalError::InvalidOperation("Unexpected 'continue' outside of loop".into()));
                }
            }
        }

        Ok(ctx.values)
    }
}

#[cfg(test)]
mod tests {
    use super::should_override_with_inferred_eos;

    #[test]
    fn inferred_eos_used_when_not_explicitly_supplied() {
        assert!(should_override_with_inferred_eos(false, Some(151645), Some(151645)));
        assert!(should_override_with_inferred_eos(false, None, Some(151645)));
    }

    #[test]
    fn inferred_eos_used_when_supplied_value_matches_workflow_default() {
        assert!(should_override_with_inferred_eos(true, Some(151645), Some(151645)));
    }

    #[test]
    fn explicit_non_default_eos_is_preserved() {
        assert!(!should_override_with_inferred_eos(true, Some(2), Some(151645)));
    }
}
