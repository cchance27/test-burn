use std::sync::Arc;

use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct TokenizeOp {
    model_id: Option<String>,
    input_var: String,
    output_var: String,
}

impl TokenizeOp {
    pub(crate) fn new(model_id: Option<String>, input_var: String, output_var: String) -> Self {
        Self {
            model_id,
            input_var,
            output_var,
        }
    }
}

impl WorkflowOp for TokenizeOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;
        let tokenizer = model.tokenizer()?;

        // Resolve input text.
        let text = if let Some(val) = ctx.values.get(&self.input_var) {
            val.as_text()
                .ok_or_else(|| MetalError::InvalidOperation(format!("TokenizeOp input variable '{}' is not text", self.input_var)))?
        } else {
            // Assume the input_var IS the literal text if not found in variables?
            // Actually, the spec should probably distinguish between literals and variables.
            // For now, let's just assume it's a variable.
            return Err(MetalError::InvalidOperation(format!(
                "TokenizeOp missing input variable '{}'",
                self.input_var
            )));
        };

        let tokens = tokenizer.encode(text)?;
        ctx.values.insert(self.output_var.clone(), Value::TokensU32(Arc::from(tokens)));

        Ok(WorkflowOpOutcome::Continue)
    }
}
