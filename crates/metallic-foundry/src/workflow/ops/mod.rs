//! Workflow op interfaces (trait objects) and implementations.

mod compute_int;
mod detokenize;
mod forward;
mod loop_op;
mod prefill;
mod ret;
mod sample;
mod set_globals;
mod sync;
mod tokenize;

pub(crate) use compute_int::ComputeIntOp;
pub(crate) use detokenize::DetokenizeOp;
pub(crate) use forward::ForwardOp;
pub(crate) use loop_op::LoopOp;
pub(crate) use prefill::PrefillOp;
pub(crate) use ret::ReturnOp;
pub(crate) use sample::SampleOp;
pub(crate) use set_globals::SetGlobalsOp;
pub(crate) use sync::SyncOp;
pub(crate) use tokenize::TokenizeOp;

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
