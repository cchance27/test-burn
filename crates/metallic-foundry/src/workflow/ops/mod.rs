//! Workflow op interfaces (trait objects) and implementations.

mod append_token;
mod check_eos;
mod compute_int;
mod control_flow;
mod detokenize;
mod format_chat;
mod forward;
mod graph_forward;

mod capture;
mod prefill;
mod ret;
mod sample;
mod set_globals;
mod stream;
mod sync;
mod tokenize;

pub(crate) use append_token::AppendTokenOp;
pub(crate) use capture::{CaptureBeginOp, CaptureEndOp, CaptureWaitOp};
pub(crate) use check_eos::CheckEosOp;
pub(crate) use compute_int::ComputeIntOp;
pub(crate) use control_flow::{BreakOp, ContinueOp, IfOp, WhileBatchedOp, WhileOp};
pub(crate) use detokenize::DetokenizeOp;
pub(crate) use format_chat::FormatChatOp;
pub(crate) use forward::ForwardOp;
pub(crate) use graph_forward::GraphForwardOp;
pub(crate) use prefill::PrefillOp;
pub(crate) use ret::ReturnOp;
pub(crate) use sample::SampleOp;
pub(crate) use set_globals::SetGlobalsOp;
pub(crate) use stream::{StreamInitOp, StreamWriteU32Op};
pub(crate) use sync::SyncOp;
pub(crate) use tokenize::TokenizeOp;

use super::runner::WorkflowExecutionContext;
use crate::error::MetalError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowOpOutcome {
    Continue,
    Return,
    Break,
    LoopContinue,
}

#[derive(Debug, Clone)]
pub struct MemoizeSpec {
    /// Workflow variable names read by this op.
    pub inputs: Vec<String>,
    /// Workflow variable names written by this op.
    pub outputs: Vec<String>,
    /// If true, memoization is disabled in interactive TUI mode.
    pub disable_in_tui: bool,
}

pub trait WorkflowOp: Send {
    /// Called once before each workflow run starts.
    ///
    /// This allows ops to reset per-run counters while keeping allocated buffers/caches.
    /// Control-flow ops should forward this call to nested ops.
    fn begin_run(&mut self, _ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        Ok(())
    }

    /// Optional memoization spec for this op.
    ///
    /// When set, the workflow engine may cache this op's outputs keyed by the hash of its declared inputs.
    /// This is intended for deterministic / CPU-bound ops (e.g. chat formatting, tokenization, encoders),
    /// not GPU-bound forward passes.
    fn memoize_spec(&self) -> Option<MemoizeSpec> {
        None
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError>;

    /// Resets any internal state of the op (e.g. counters, history).
    /// Called when the workflow context/session is reset (e.g. switching conversations).
    fn reset(&mut self) {}
}
