use std::sync::Arc;

use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct DetokenizeOp {
    model_id: Option<String>,
    input_var: String,
    output_var: String,
}

impl DetokenizeOp {
    pub(crate) fn new(model_id: Option<String>, input_var: String, output_var: String) -> Self {
        Self {
            model_id,
            input_var,
            output_var,
        }
    }
}

impl WorkflowOp for DetokenizeOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;
        let tokenizer = model.tokenizer()?;

        // Resolve input tokens.
        let val = ctx
            .values
            .get(&self.input_var)
            .ok_or_else(|| MetalError::InvalidOperation(format!("DetokenizeOp missing input variable '{}'", self.input_var)))?;

        let text = match val {
            Value::U32(id) => tokenizer.decode(&[*id])?,
            Value::TokensU32(tokens) => tokenizer.decode(tokens)?,
            _ => {
                return Err(MetalError::InvalidOperation(format!(
                    "DetokenizeOp input variable '{}' is not u32 or list of tokens",
                    self.input_var
                )));
            }
        };

        ctx.values.insert(self.output_var.clone(), Value::Text(Arc::from(text)));

        Ok(WorkflowOpOutcome::Continue)
    }
}
