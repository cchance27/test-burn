use crate::{
    error::MetalError, workflow::{
        ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct ReturnOp {
    output: String,
}

impl ReturnOp {
    pub(crate) fn new(output: String) -> Self {
        Self { output }
    }
}

impl WorkflowOp for ReturnOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        if !ctx.values.contains_key(&self.output) {
            return Err(MetalError::InvalidOperation(format!(
                "workflow return references missing output value '{}'",
                self.output
            )));
        }
        ctx.return_key = Some(self.output.clone());
        Ok(WorkflowOpOutcome::Return)
    }
}
