use std::sync::{OnceLock, RwLock};

use metallic_env::{FoundryEnvVar, is_set};
use metallic_sdk::debug::op_metrics_enabled;
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;

use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{
        AppendTokenOp, BreakOp, CaptureBeginOp, CaptureEndOp, CaptureWaitOp, CheckEosOp, ComputeIntOp, ContinueOp, DetokenizeOp, FormatChatOp, ForwardOp, GraphForwardOp, IfOp, MemoizeSpec, PrefillOp, ReturnOp, SampleOp, SetGlobalsOp, StreamInitOp, StreamWriteU32Op, SyncOp, TokenizeOp, WhileBatchedOp, WhileOp, WorkflowOp
    }
};
use crate::error::MetalError;

pub(super) struct CompiledWorkflow {
    pub(super) ops: Vec<CompiledOp>,
}

pub(super) struct CompiledOp {
    pub(super) name: String,
    pub(super) op: Box<dyn WorkflowOp>,
}

struct MemoizedOp {
    op_name: String,
    inner: Box<dyn WorkflowOp>,
    spec: MemoizeSpec,
    cached: Option<(u64, rustc_hash::FxHashMap<String, super::Value>)>,
}

struct TimedOp {
    op_name: String,
    inner: Box<dyn WorkflowOp>,
}

impl TimedOp {
    fn record_latency(&self, ctx: &mut super::runner::WorkflowExecutionContext<'_>, elapsed_us: usize) {
        let metrics_val = ctx
            .values
            .entry("_internal.op_metrics".to_string())
            .or_insert_with(|| super::Value::Map(rustc_hash::FxHashMap::default()));

        let metrics_map = match metrics_val {
            super::Value::Map(map) => map,
            other => {
                *other = super::Value::Map(rustc_hash::FxHashMap::default());
                match other {
                    super::Value::Map(map) => map,
                    _ => return,
                }
            }
        };

        let op_val = metrics_map
            .entry(self.op_name.clone())
            .or_insert_with(|| super::Value::Map(rustc_hash::FxHashMap::default()));
        let op_map = match op_val {
            super::Value::Map(map) => map,
            other => {
                *other = super::Value::Map(rustc_hash::FxHashMap::default());
                match other {
                    super::Value::Map(map) => map,
                    _ => return,
                }
            }
        };

        let count = op_map.get("count").and_then(super::Value::as_usize).unwrap_or(0).saturating_add(1);
        let total_us = op_map
            .get("total_us")
            .and_then(super::Value::as_usize)
            .unwrap_or(0)
            .saturating_add(elapsed_us);
        let max_us = op_map.get("max_us").and_then(super::Value::as_usize).unwrap_or(0).max(elapsed_us);

        op_map.insert("count".to_string(), super::Value::Usize(count));
        op_map.insert("total_us".to_string(), super::Value::Usize(total_us));
        op_map.insert("max_us".to_string(), super::Value::Usize(max_us));
        op_map.insert("last_us".to_string(), super::Value::Usize(elapsed_us));
    }
}

impl MemoizedOp {
    fn key_for(&self, ctx: &super::runner::WorkflowExecutionContext<'_>) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        self.op_name.hash(&mut h);

        // Include declared input values.
        for name in &self.spec.inputs {
            let v = ctx.values.get(name)?;
            v.fingerprint64().hash(&mut h);
        }
        Some(h.finish())
    }
}

impl WorkflowOp for MemoizedOp {
    fn begin_run(&mut self, ctx: &mut super::runner::WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        self.inner.begin_run(ctx)
    }

    fn memoize_spec(&self) -> Option<MemoizeSpec> {
        // Wrapper handles memoization; don't nest.
        None
    }

    fn execute(
        &mut self,
        ctx: &mut super::runner::WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<super::ops::WorkflowOpOutcome, MetalError> {
        if self.spec.disable_in_tui && is_set(FoundryEnvVar::TuiMode) {
            return self.inner.execute(ctx, on_token);
        }

        let Some(key) = self.key_for(ctx) else {
            return self.inner.execute(ctx, on_token);
        };

        if let Some((cached_key, cached_vals)) = &self.cached
            && *cached_key == key
        {
            for (k, v) in cached_vals {
                ctx.values.insert(k.clone(), v.clone());
            }
            return Ok(super::ops::WorkflowOpOutcome::Continue);
        }

        let out = self.inner.execute(ctx, on_token)?;
        if matches!(out, super::ops::WorkflowOpOutcome::Continue) {
            let mut captured: rustc_hash::FxHashMap<String, super::Value> = rustc_hash::FxHashMap::default();
            for name in &self.spec.outputs {
                if let Some(v) = ctx.values.get(name) {
                    captured.insert(name.clone(), v.clone());
                }
            }
            self.cached = Some((key, captured));
        }
        Ok(out)
    }

    fn reset(&mut self) {
        self.cached = None;
        self.inner.reset();
    }
}

impl WorkflowOp for TimedOp {
    fn begin_run(&mut self, ctx: &mut super::runner::WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        self.inner.begin_run(ctx)
    }

    fn memoize_spec(&self) -> Option<MemoizeSpec> {
        // Preserve memoization behavior of the wrapped op.
        self.inner.memoize_spec()
    }

    fn execute(
        &mut self,
        ctx: &mut super::runner::WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<super::ops::WorkflowOpOutcome, MetalError> {
        let start = std::time::Instant::now();
        let out = self.inner.execute(ctx, on_token);
        self.record_latency(ctx, start.elapsed().as_micros() as usize);
        out
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// A builder function for creating a `WorkflowOp` from a JSON value.
pub type OpBuilder = Box<dyn Fn(serde_json::Value) -> Result<Box<dyn WorkflowOp>, MetalError> + Send + Sync>;

/// A global registry for workflow operations.
/// Usage: `REGISTRY.get().unwrap().read().unwrap().get("op_name")`
fn get_registry() -> &'static RwLock<FxHashMap<String, OpBuilder>> {
    static REGISTRY: OnceLock<RwLock<FxHashMap<String, OpBuilder>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m = FxHashMap::default();
        initialize_standard_ops(&mut m);
        RwLock::new(m)
    })
}

/// Registers a new operation into the global registry.
pub fn register_op<F>(name: impl Into<String>, builder: F)
where
    F: Fn(serde_json::Value) -> Result<Box<dyn WorkflowOp>, MetalError> + Send + Sync + 'static,
{
    let registry = get_registry();
    let mut write_guard = registry.write().unwrap();
    write_guard.insert(name.into(), Box::new(builder));
}

/// Helper function to create a builder for a standard operation that takes a `Spec` struct.
fn make_std_builder<S, O, C>(ctor: C) -> OpBuilder
where
    S: DeserializeOwned,
    O: WorkflowOp + 'static,
    C: Fn(S) -> O + Send + Sync + 'static,
{
    Box::new(move |v| {
        let spec: S = serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid spec for op: {}", e)))?;
        Ok(Box::new(ctor(spec)))
    })
}

fn initialize_standard_ops(m: &mut FxHashMap<String, OpBuilder>) {
    // Standard leaf operations
    m.insert("prefill".to_string(), make_std_builder(PrefillOp::new));
    m.insert("forward".to_string(), make_std_builder(ForwardOp::new));
    m.insert("sample".to_string(), make_std_builder(SampleOp::new));
    m.insert("tokenize".to_string(), make_std_builder(TokenizeOp::new));
    m.insert("detokenize".to_string(), make_std_builder(DetokenizeOp::new));
    m.insert("format_chat".to_string(), make_std_builder(FormatChatOp::new));
    m.insert("set_globals".to_string(), make_std_builder(SetGlobalsOp::new));
    m.insert("return".to_string(), make_std_builder(ReturnOp::new));
    m.insert("compute_int".to_string(), make_std_builder(ComputeIntOp::new));
    m.insert("check_eos".to_string(), make_std_builder(CheckEosOp::new));
    m.insert("append_token".to_string(), make_std_builder(AppendTokenOp::new));
    m.insert("capture_begin".to_string(), make_std_builder(CaptureBeginOp::new));
    m.insert("capture_end".to_string(), make_std_builder(CaptureEndOp::new));
    m.insert("capture_wait".to_string(), make_std_builder(CaptureWaitOp::new));
    m.insert("stream_init".to_string(), make_std_builder(StreamInitOp::new));
    m.insert("stream_write_u32".to_string(), make_std_builder(StreamWriteU32Op::new));
    m.insert(
        "graph_forward".to_string(),
        Box::new(|v| {
            let spec: super::spec::GraphForwardSpec =
                serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid spec for op: {}", e)))?;
            // DX guardrail: `logits_binding` is both the model binding name and the workflow variable name.
            // Most model specs expose this as "logits"; mismatches fail at runtime with confusing "missing binding".
            if spec.logits_binding != "logits" {
                return Err(MetalError::InvalidOperation(
                    "graph_forward.logits_binding must be 'logits' (model binding name)".into(),
                ));
            }
            Ok(Box::new(GraphForwardOp::new(spec)))
        }),
    );

    // Unit operations
    m.insert("synchronize".to_string(), Box::new(|_| Ok(Box::new(SyncOp))));
    m.insert("break".to_string(), Box::new(|_| Ok(Box::new(BreakOp))));
    m.insert("continue".to_string(), Box::new(|_| Ok(Box::new(ContinueOp))));

    // Recursive operations need manual implementation to access compile_steps
    m.insert(
        "if".to_string(),
        Box::new(|v| {
            let spec: super::spec::IfSpec =
                serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid if spec: {}", e)))?;
            let then_ops = compile_steps(&spec.then)?.into_iter().map(|c| c.op).collect();
            let else_ops = compile_steps(&spec.else_)?.into_iter().map(|c| c.op).collect();
            Ok(Box::new(IfOp::new(spec.condition, then_ops, else_ops)))
        }),
    );

    m.insert(
        "while".to_string(),
        Box::new(|v| {
            let spec: super::spec::WhileSpec =
                serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid while spec: {}", e)))?;
            let body_ops = compile_steps(&spec.body)?.into_iter().map(|c| c.op).collect();
            Ok(Box::new(WhileOp::new(spec.condition, spec.max_iterations, body_ops)))
        }),
    );

    m.insert(
        "while_batched".to_string(),
        Box::new(|v| {
            let spec: super::spec::WhileBatchedSpec =
                serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid while_batched spec: {}", e)))?;
            let body_ops = compile_steps(&spec.body)?.into_iter().map(|c| c.op).collect();
            Ok(Box::new(WhileBatchedOp::new(
                spec.condition,
                spec.max_iterations,
                spec.batch_size,
                spec.unsafe_allow_overshoot,
                spec.token_var,
                spec.stream_channel,
                spec.stream_async_poll,
                spec.stream_poll_interval_us,
                spec.output_tokens,
                spec.eos_token,
                body_ops,
            )))
        }),
    );
}

fn compile_steps(steps: &[WorkflowStepSpec]) -> Result<Vec<CompiledOp>, MetalError> {
    let mut ops: Vec<CompiledOp> = Vec::with_capacity(steps.len());
    let registry = get_registry().read().unwrap();
    let enable_timing = op_metrics_enabled();

    for step in steps {
        if let Some(builder) = registry.get(step.op.as_str()) {
            let op = builder(step.params.clone())?;
            let op: Box<dyn WorkflowOp> = if let Some(spec) = op.memoize_spec() {
                Box::new(MemoizedOp {
                    op_name: step.op.clone(),
                    inner: op,
                    spec,
                    cached: None,
                })
            } else {
                op
            };
            // Wrap every op uniformly so latency accounting is centralized and consistent.
            let op: Box<dyn WorkflowOp> = if enable_timing {
                Box::new(TimedOp {
                    op_name: step.op.clone(),
                    inner: op,
                })
            } else {
                op
            };
            ops.push(CompiledOp { name: step.op.clone(), op });
        } else {
            return Err(MetalError::InvalidOperation(format!("Unknown workflow op: {}", step.op)));
        }
    }

    Ok(ops)
}

impl CompiledWorkflow {
    pub(super) fn compile(workflow: &WorkflowSpec) -> Result<Self, MetalError> {
        let ops = compile_steps(&workflow.steps)?;
        Ok(Self { ops })
    }

    pub(super) fn reset(&mut self) {
        for op in &mut self.ops {
            op.op.reset();
        }
    }
}
