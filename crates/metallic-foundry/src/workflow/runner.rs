use rustc_hash::FxHashMap;

use super::{
    Value, compiler::CompiledWorkflow, ops::WorkflowOpOutcome, spec::{Param, WorkflowSpec}
};
use crate::{Foundry, error::MetalError, model::CompiledModel};

pub struct WorkflowRunnerConfig {
    pub workflow: WorkflowSpec,
}

pub(crate) struct WorkflowExecutionContext<'a> {
    pub(crate) workflow: &'a WorkflowSpec,
    pub(crate) foundry: &'a mut Foundry,
    pub(crate) models: &'a FxHashMap<String, &'a CompiledModel>,
    pub(crate) values: FxHashMap<String, Value>,
    pub(crate) return_key: Option<String>,
}

impl<'a> WorkflowExecutionContext<'a> {
    pub(crate) fn resolve_model(&self, model_id: Option<&str>) -> Result<&'a CompiledModel, MetalError> {
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
            .copied()
            .ok_or_else(|| MetalError::InvalidOperation(format!("Workflow references unknown model_id '{id}'")))
    }

    pub(crate) fn resolve_param_u32(&self, p: &Param<u32>) -> Result<u32, MetalError> {
        match p {
            Param::Literal(v) => Ok(*v),
            Param::Input(name) => self
                .values
                .get(name)
                .and_then(|v| match v {
                    Value::U32(v) => Some(*v),
                    Value::Usize(v) => (*v).try_into().ok(),
                    _ => None,
                })
                .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{name}' (u32)"))),
        }
    }

    pub(crate) fn resolve_param_f32(&self, p: &Param<f32>) -> Result<f32, MetalError> {
        match p {
            Param::Literal(v) => Ok(*v),
            Param::Input(name) => self
                .values
                .get(name)
                .and_then(|v| match v {
                    Value::F32(v) => Some(*v),
                    _ => None,
                })
                .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{name}' (f32)"))),
        }
    }

    pub(crate) fn read_usize(&self, name: &str) -> Result<usize, MetalError> {
        self.values
            .get(name)
            .and_then(|v| match v {
                Value::Usize(v) => Some(*v),
                Value::U32(v) => Some(*v as usize),
                _ => None,
            })
            .ok_or_else(|| MetalError::InvalidOperation(format!("Missing workflow input '{name}' (usize/u32)")))
    }
}

pub struct WorkflowRunner<'a> {
    foundry: &'a mut Foundry,
    models: FxHashMap<String, &'a CompiledModel>,
}

impl<'a> WorkflowRunner<'a> {
    pub fn new(foundry: &'a mut Foundry, models: FxHashMap<String, &'a CompiledModel>) -> Self {
        Self { foundry, models }
    }

    pub fn run_streaming(
        &mut self,
        cfg: &WorkflowRunnerConfig,
        mut inputs: FxHashMap<String, Value>,
        mut on_token: impl FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<FxHashMap<String, Value>, MetalError> {
        // Apply defaults from workflow inputs.
        for inp in &cfg.workflow.inputs {
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
                "f32" => {
                    if let Some(v) = default.as_f64() {
                        inputs.insert(inp.name.clone(), Value::F32(v as f32));
                    }
                }
                _ => {}
            }
        }

        let compiled = CompiledWorkflow::compile(&cfg.workflow)?;

        let mut ctx = WorkflowExecutionContext {
            workflow: &cfg.workflow,
            foundry: self.foundry,
            models: &self.models,
            values: inputs,
            return_key: None,
        };

        for mut op in compiled.ops {
            match op.execute(&mut ctx, &mut on_token)? {
                WorkflowOpOutcome::Continue => {}
                WorkflowOpOutcome::Return => break,
            }
        }

        Ok(ctx.values)
    }
}
