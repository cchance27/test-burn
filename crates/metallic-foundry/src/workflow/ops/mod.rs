//! Workflow op interfaces (trait objects) and implementations.

mod append_token;
mod check_eos;
mod compute_int;
mod control_flow;
mod detokenize;
mod forward;
mod graph_forward;

mod prefill;
mod ret;
mod sample;
mod set_globals;
mod sync;
mod tokenize;

pub(crate) use append_token::AppendTokenOp;
pub(crate) use check_eos::CheckEosOp;
pub(crate) use compute_int::ComputeIntOp;
pub(crate) use control_flow::{BreakOp, ContinueOp, IfOp, WhileOp};
pub(crate) use detokenize::DetokenizeOp;
pub(crate) use forward::ForwardOp;
pub(crate) use graph_forward::GraphForwardOp;
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
    Break,
    LoopContinue,
}

pub(crate) trait WorkflowOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError>;
}
