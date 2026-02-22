use crate::{
    error::MetalError, workflow::{
        Value, ops::{
            WorkflowOp, WorkflowOpOutcome, common::{INTERNAL_LAST_DECODE_US, callback_timings_from_ctx, err_invalid_input_type, err_missing_input, read_internal_usize}
        }, runner::WorkflowExecutionContext
    }
};

pub(crate) struct AppendTokenOp {
    input: String,
    output: String,
}

impl AppendTokenOp {
    pub(crate) fn new(spec: crate::workflow::spec::AppendTokenSpec) -> Self {
        Self {
            input: spec.input,
            output: spec.output,
        }
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
            .ok_or_else(|| err_missing_input("AppendTokenOp", &self.input, "u32"))?;

        // Append to output list
        if let Some(val) = ctx.values.get_mut(&self.output) {
            if let Value::TokensU32(vec) = val {
                vec.push(token);
            } else {
                return Err(err_invalid_input_type("AppendTokenOp", &self.output, "TokensU32"));
            }
        } else {
            ctx.values.insert(self.output.clone(), Value::TokensU32(vec![token]));
        }

        // Call on_token callback
        let decode_us = read_internal_usize(ctx, INTERNAL_LAST_DECODE_US);
        let (prefill_dur, setup_dur, decode_dur) = callback_timings_from_ctx(ctx, Some(decode_us));

        let should_continue = on_token(token, prefill_dur, setup_dur, decode_dur)?;

        if !should_continue {
            return Ok(WorkflowOpOutcome::Break);
            // NOTE: If we are in a loop, Break is correct. If top level, it might need Return?
            // But usually this op is inside a loop.
            // If the user wants to stop generation, we break the loop.
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}
