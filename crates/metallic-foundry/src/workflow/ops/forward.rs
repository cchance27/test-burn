use rustc_hash::FxHashMap;

use crate::{
    Foundry, error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct ForwardOp {
    model_id: Option<String>,
    inputs: FxHashMap<String, String>,
    outputs: FxHashMap<String, String>,
    update_globals: FxHashMap<String, Param<usize>>,
    apply_derived_globals: bool,
    #[allow(dead_code)]
    description: Option<String>,
}

impl ForwardOp {
    pub(crate) fn new(spec: crate::workflow::spec::ForwardSpec) -> Self {
        Self {
            model_id: spec.model_id,
            inputs: spec.inputs,
            outputs: spec.outputs,
            update_globals: spec.update_globals,
            apply_derived_globals: spec.apply_derived_globals,
            description: spec.description,
        }
    }
}

impl WorkflowOp for ForwardOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;

        // 1. Resolve globals and inputs
        let mut resolved_globals = Vec::new();
        for (name, param) in &self.update_globals {
            resolved_globals.push((name.clone(), ctx.resolve_param_usize(param)?));
        }

        let mut resolved_inputs = Vec::new();
        for (binding_name, var_name) in &self.inputs {
            let value = ctx
                .values
                .get(var_name)
                .ok_or_else(|| MetalError::InvalidOperation(format!("ForwardOp missing input variable '{}'", var_name)))?;
            resolved_inputs.push((binding_name.clone(), value.clone()));
        }

        // 2. Execute forward pass
        let mut extracted_outputs = Vec::new();
        model.with_session_mut(ctx.foundry, |foundry: &mut Foundry, session| {
            let bindings = &mut session.bindings;
            let fast_bindings = &mut session.fast_bindings;

            // Update globals
            for (name, val) in resolved_globals {
                model.set_int_global(bindings, &name, val);
            }
            if self.apply_derived_globals {
                model.apply_derived_globals(bindings);
            }

            // Bind inputs
            for (binding_name, value) in resolved_inputs {
                match value {
                    Value::TokensU32(tokens) => {
                        let arg = bindings.get(&binding_name)?;
                        let buffer = arg.buffer.as_ref().ok_or_else(|| {
                            MetalError::InvalidOperation(format!("ForwardOp input binding '{}' is not a buffer", binding_name))
                        })?;
                        buffer.copy_from_slice(&tokens);
                    }
                    Value::Tensor(arg) => {
                        model.set_binding(bindings, fast_bindings, &binding_name, arg.clone());
                    }
                    _ => {
                        return Err(MetalError::InvalidOperation(format!(
                            "ForwardOp cannot bind value to model input '{}'",
                            binding_name
                        )));
                    }
                }
            }

            model.forward(foundry, bindings, fast_bindings)?;

            // Extract outputs
            for (binding_name, var_name) in &self.outputs {
                let arg = bindings.get(binding_name)?;
                extracted_outputs.push((var_name.clone(), arg.clone()));
            }

            Ok::<(), MetalError>(())
        })?;

        // 3. Update ctx values
        for (var_name, arg) in extracted_outputs {
            ctx.values.insert(var_name, Value::Tensor(arg));
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}
