use crate::{
    Foundry, error::MetalError, types::{MetalBuffer, MetalResourceOptions, TensorArg}, workflow::{
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
    input_token_buffer: Option<MetalBuffer>,
    input_token_arg: Option<TensorArg>,
}

impl GraphForwardOp {
    pub(crate) fn new(spec: crate::workflow::spec::GraphForwardSpec) -> Self {
        Self {
            model_id: spec.model_id,
            token_var: spec.token_var,
            input_ids_binding: spec.input_ids_binding.unwrap_or_else(|| "input_ids".to_string()),
            logits_binding: spec.logits_binding,
            position_offset_key: spec.position_offset_key.unwrap_or_else(|| "position_offset".to_string()),
            position: spec.position,
            apply_derived_globals: spec.apply_derived_globals,
            description: spec.description,
            input_token_buffer: None,
            input_token_arg: None,
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
        let token_u32 = ctx.values.get(&self.token_var).and_then(|v| v.as_u32());
        let token_tensor = ctx.values.get(&self.token_var).and_then(|v| v.as_tensor());
        if token_u32.is_none() && token_tensor.is_none() {
            return Err(MetalError::InvalidOperation(format!(
                "GraphForwardOp missing token variable '{}' (u32 or Tensor u32[1])",
                self.token_var
            )));
        }

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

            // Ensure KV capacity for this step. Without this, decode can write past the KV buffers or
            // trigger growth without preserving history, which manifests as severe repetition.
            model.ensure_kv_capacity(
                foundry,
                bindings,
                fast_bindings,
                &mut session.context_config,
                session.current_pos,
                pos_val.saturating_add(1),
            )?;

            {
                if let Some(token_arg) = token_tensor {
                    // Fast path: token already lives in a GPU-visible buffer (e.g. SampleOp output).
                    // Bind it directly to avoid a CPU copy + scalar readback.
                    model.set_binding(bindings, fast_bindings, &self.input_ids_binding, token_arg.clone());
                } else if let Some(token) = token_u32 {
                    // Fallback: write a scalar token into a persistent 1-token buffer.
                    if self.input_token_buffer.is_none() {
                        let buf = foundry
                            .device
                            .new_buffer(4, MetalResourceOptions::StorageModeShared)
                            .expect("Failed to allocate graph_forward input token buffer");
                        let arg = TensorArg::from_buffer(buf.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
                        self.input_token_buffer = Some(buf);
                        self.input_token_arg = Some(arg);
                    }
                    let buf = self.input_token_buffer.as_ref().expect("input_token_buffer set");
                    buf.copy_from_slice(&[token]);
                    let input_tensor = self.input_token_arg.as_ref().expect("input_token_arg set").clone();
                    model.set_binding(bindings, fast_bindings, &self.input_ids_binding, input_tensor);
                }
            }

            model.set_int_global(bindings, &self.position_offset_key, pos_val);
            if self.apply_derived_globals {
                model.apply_derived_globals(bindings);
            }

            model.forward(foundry, bindings, fast_bindings)?;

            // Keep session position consistent with the workflow's position variable so KV growth
            // can preserve the correct number of tokens.
            session.current_pos = pos_val.saturating_add(1);

            let logits = bindings.get(&self.logits_binding)?.clone();
            Ok(logits)
        })?;

        ctx.values.insert(self.logits_binding.clone(), Value::Tensor(logits_arg));

        Ok(WorkflowOpOutcome::Continue)
    }
}
