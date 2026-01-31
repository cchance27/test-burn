use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct AppendTokenOp {
    input: String,
    output: String,
}

impl AppendTokenOp {
    pub(crate) fn new(input: String, output: String) -> Self {
        Self { input, output }
    }
}

impl WorkflowOp for AppendTokenOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let token = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("AppendTokenOp missing input token '{}' (u32)", self.input)))?;

        // Append to output list
        if let Some(val) = ctx.values.get_mut(&self.output) {
            if let Value::TokensU32(vec) = val {
                vec.push(token);
            } else {
                return Err(MetalError::InvalidOperation(format!(
                    "AppendTokenOp output '{}' is not a TokensU32 list",
                    self.output
                )));
            }
        } else {
            ctx.values.insert(self.output.clone(), Value::TokensU32(vec![token]));
        }

        // Call on_token callback
        // We attempt to read metrics from internal variables if they exist, otherwise 0
        let prefill_us = ctx.read_usize("_internal.prefill_us").unwrap_or(0);
        let setup_us = ctx.read_usize("_internal.setup_us").unwrap_or(0);
        // Iteration duration is harder to track generically without a specific timer op context.
        // Passing None for now.

        let should_continue = on_token(
            token,
            std::time::Duration::from_micros(prefill_us as u64),
            std::time::Duration::from_micros(setup_us as u64),
            None,
        )?;

        if !should_continue {
            return Ok(WorkflowOpOutcome::Break);
            // NOTE: If we are in a loop, Break is correct. If top level, it might need Return?
            // But usually this op is inside a loop.
            // If the user wants to stop generation, we break the loop.
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}
