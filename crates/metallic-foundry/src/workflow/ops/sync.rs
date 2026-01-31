use crate::{
    error::MetalError, workflow::{
        ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct SyncOp;

impl WorkflowOp for SyncOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        ctx.foundry.synchronize()?;
        Ok(WorkflowOpOutcome::Continue)
    }
}
