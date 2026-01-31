use crate::{
    Foundry, error::MetalError, types::TensorArg, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct GraphForwardOp {
    model_id: Option<String>,
    token_var: String,
    input_ids_binding: String,
    logits_binding: String,
    position_offset_key: String,
    position: Option<Param<usize>>,
    apply_derived_globals: bool,
    #[allow(dead_code)]
    description: Option<String>,
}

impl GraphForwardOp {
    pub(crate) fn new(
        model_id: Option<String>,
        token_var: String,
        input_ids_binding: Option<String>,
        logits_binding: String,
        position_offset_key: Option<String>,
        position: Option<Param<usize>>,
        apply_derived_globals: bool,
        description: Option<String>,
    ) -> Self {
        Self {
            model_id,
            token_var,
            input_ids_binding: input_ids_binding.unwrap_or_else(|| "input_ids".to_string()),
            logits_binding,
            position_offset_key: position_offset_key.unwrap_or_else(|| "position_offset".to_string()),
            position,
            apply_derived_globals,
            description,
        }
    }
}

impl WorkflowOp for GraphForwardOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;

        // Resolve generic inputs
        let token = ctx
            .values
            .get(&self.token_var)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("GraphForwardOp missing token variable '{}' (u32)", self.token_var)))?;

        let pos_val = if let Some(p) = &self.position {
            ctx.resolve_param_usize(p)?
        } else {
            // If not provided, maybe we can assume 0? But that's dangerous.
            // Or maybe we can't run.
            return Err(MetalError::InvalidOperation(
                "GraphForwardOp requires 'position' parameter in generic workflow".into(),
            ));
        };

        let logits_arg = model.with_session_mut(ctx.foundry, |foundry: &mut Foundry, session| {
            let bindings = &mut session.bindings;
            let fast_bindings = &mut session.fast_bindings;

            {
                // Create a temporary buffer for the input token.

                let input_buffer = foundry
                    .device
                    .new_buffer_from_slice(&[token], crate::types::MetalResourceOptions::StorageModeManaged)
                    .unwrap();
                let input_tensor = TensorArg::from_buffer(input_buffer, crate::tensor::Dtype::U32, vec![1], vec![1]);

                model.set_binding(bindings, fast_bindings, &self.input_ids_binding, input_tensor);
            }

            model.set_int_global(bindings, &self.position_offset_key, pos_val);
            if self.apply_derived_globals {
                model.apply_derived_globals(bindings);
            }

            model.forward(foundry, bindings, fast_bindings)?;

            let logits = bindings.get(&self.logits_binding)?.clone();
            Ok(logits)
        })?;

        ctx.values.insert(self.logits_binding.clone(), Value::Tensor(logits_arg));

        Ok(WorkflowOpOutcome::Continue)
    }
}
