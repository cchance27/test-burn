use std::sync::{OnceLock, RwLock};

use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;

use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{
        AppendTokenOp, BreakOp, CaptureBeginOp, CaptureEndOp, CaptureWaitOp, CheckEosOp, ComputeIntOp, ContinueOp, DetokenizeOp, FormatChatOp, ForwardOp, GraphForwardOp, IfOp, MemoizeSpec, PrefillOp, ReturnOp, SampleOp, SetGlobalsOp, StreamInitOp, StreamWriteU32Op, SyncOp, TokenizeOp, WhileBatchedOp, WhileOp, WorkflowOp
    }
};
use crate::error::MetalError;

pub(crate) struct CompiledWorkflow {
    pub(crate) ops: Vec<CompiledOp>,
}

pub(crate) struct CompiledOp {
    pub name: String,
    pub op: Box<dyn WorkflowOp>,
}

struct MemoizedOp {
    op_name: String,
    inner: Box<dyn WorkflowOp>,
    spec: MemoizeSpec,
    cached: Option<(u64, rustc_hash::FxHashMap<String, super::Value>)>,
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
        if self.spec.disable_in_tui && std::env::var("METALLIC_TUI_MODE").is_ok() {
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
#[allow(dead_code)]
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

    for step in steps {
        if let Some(builder) = registry.get(step.op.as_str()) {
            let op = builder(step.params.clone())?;
            if let Some(spec) = op.memoize_spec() {
                ops.push(CompiledOp {
                    name: step.op.clone(),
                    op: Box::new(MemoizedOp {
                        op_name: step.op.clone(),
                        inner: op,
                        spec,
                        cached: None,
                    }),
                });
            } else {
                ops.push(CompiledOp { name: step.op.clone(), op });
            }
        } else {
            return Err(MetalError::InvalidOperation(format!("Unknown workflow op: {}", step.op)));
        }
    }

    Ok(ops)
}

impl CompiledWorkflow {
    pub(crate) fn compile(workflow: &WorkflowSpec) -> Result<Self, MetalError> {
        let ops = compile_steps(&workflow.steps)?;
        Ok(Self { ops })
    }
}
