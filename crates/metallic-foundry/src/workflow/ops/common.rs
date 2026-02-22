use std::time::Duration;

use super::WorkflowOp;
use crate::{
    error::MetalError, workflow::{Value, runner::WorkflowExecutionContext}
};

pub(super) const INTERNAL_PREFIX: &str = "_internal.";
pub(super) const INTERNAL_KV_PREFIX_KEY: &str = "_internal.kv_prefix_key";
pub(super) const INTERNAL_KV_PREFIX_BASE_KEY: &str = "_internal.kv_prefix_base_key";
pub(super) const INTERNAL_START_POS: &str = "_internal.start_pos";
pub(super) const INTERNAL_PROMPT_LEN: &str = "_internal.prompt_len";
pub(super) const INTERNAL_END_POS: &str = "_internal.end_pos";
pub(super) const INTERNAL_LAST_PREFILL_M: &str = "_internal.last_prefill_m";
pub(super) const INTERNAL_PREFILL_US: &str = "_internal.prefill_us";
pub(super) const INTERNAL_SETUP_US: &str = "_internal.setup_us";
pub(super) const INTERNAL_LAST_DECODE_US: &str = "_internal.last_decode_us";
pub(super) const INTERNAL_DECODE_BATCH_SIZE: &str = "_internal.decode_batch_size";
pub(super) const INTERNAL_DECODE_BATCH_IDX: &str = "_internal.decode_batch_idx";

#[inline]
pub(super) fn err_missing_input(op: &str, name: &str, expected: &str) -> MetalError {
    MetalError::InvalidOperation(format!("{op} missing input '{name}' ({expected})"))
}

#[inline]
pub(super) fn err_invalid_input_type(op: &str, name: &str, expected: &str) -> MetalError {
    MetalError::InvalidOperation(format!("{op} input '{name}' is not {expected}"))
}

#[inline]
pub(super) fn read_internal_usize(ctx: &WorkflowExecutionContext<'_>, key: &str) -> usize {
    ctx.read_usize(key).unwrap_or(0)
}

#[inline]
pub(super) fn write_internal_usize(ctx: &mut WorkflowExecutionContext<'_>, key: &'static str, value: usize) {
    debug_assert!(key.starts_with(INTERNAL_PREFIX));
    ctx.values.insert(key.to_string(), Value::Usize(value));
}

#[inline]
pub(super) fn remove_internal(ctx: &mut WorkflowExecutionContext<'_>, key: &'static str) {
    debug_assert!(key.starts_with(INTERNAL_PREFIX));
    ctx.values.remove(key);
}

pub(super) fn condition_as_bool(ctx: &WorkflowExecutionContext<'_>, op: &str, condition_var: &str) -> Result<bool, MetalError> {
    let cond_val = ctx
        .values
        .get(condition_var)
        .ok_or_else(|| err_missing_input(op, condition_var, "bool|u32|usize"))?;
    match cond_val {
        Value::Bool(b) => Ok(*b),
        Value::U32(v) => Ok(*v != 0),
        Value::Usize(v) => Ok(*v != 0),
        _ => Err(err_invalid_input_type(op, condition_var, "bool|u32|usize")),
    }
}

pub(super) fn begin_run_nested(ops: &mut [Box<dyn WorkflowOp>], ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
    for op in ops {
        op.begin_run(ctx)?;
    }
    Ok(())
}

pub(super) fn reset_nested(ops: &mut [Box<dyn WorkflowOp>]) {
    for op in ops {
        op.reset();
    }
}

#[inline]
pub(super) fn callback_timings_from_ctx(
    ctx: &WorkflowExecutionContext<'_>,
    decode_us: Option<usize>,
) -> (Duration, Duration, Option<Duration>) {
    let prefill_us = read_internal_usize(ctx, INTERNAL_PREFILL_US);
    let setup_us = read_internal_usize(ctx, INTERNAL_SETUP_US);
    let decode = decode_us.and_then(|us| (us > 0).then_some(Duration::from_micros(us as u64)));
    (
        Duration::from_micros(prefill_us as u64),
        Duration::from_micros(setup_us as u64),
        decode,
    )
}
