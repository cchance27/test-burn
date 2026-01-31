//! Workflow op interfaces (trait objects) and implementations.

mod loop_op;
mod prefill;
mod ret;
mod set_globals;
mod sync;

pub(crate) use loop_op::LoopOp;
pub(crate) use prefill::PrefillOp;
pub(crate) use ret::ReturnOp;
pub(crate) use set_globals::SetGlobalsOp;
pub(crate) use sync::SyncOp;

use super::runner::WorkflowExecutionContext;
use crate::error::MetalError;

pub(crate) enum WorkflowOpOutcome {
    Continue,
    Return,
}

pub(crate) trait WorkflowOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError>;
}
