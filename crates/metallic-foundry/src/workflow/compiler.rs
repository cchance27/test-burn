use std::sync::{OnceLock, RwLock};

use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;

use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{
        AppendTokenOp, BreakOp, CheckEosOp, ComputeIntOp, ContinueOp, DetokenizeOp, FormatChatOp, ForwardOp, GraphForwardOp, IfOp, PrefillOp, ReturnOp, SampleOp, SetGlobalsOp, SyncOp, TokenizeOp, WhileOp, WorkflowOp
    }
};
use crate::error::MetalError;

pub(crate) struct CompiledWorkflow {
    pub(crate) ops: Vec<Box<dyn WorkflowOp>>,
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
    m.insert("graph_forward".to_string(), make_std_builder(GraphForwardOp::new));

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
            Ok(Box::new(IfOp::new(
                spec.condition,
                compile_steps(&spec.then)?,
                compile_steps(&spec.else_)?,
            )))
        }),
    );

    m.insert(
        "while".to_string(),
        Box::new(|v| {
            let spec: super::spec::WhileSpec =
                serde_json::from_value(v).map_err(|e| MetalError::InvalidOperation(format!("Invalid while spec: {}", e)))?;
            Ok(Box::new(WhileOp::new(
                spec.condition,
                spec.max_iterations,
                compile_steps(&spec.body)?,
            )))
        }),
    );
}

fn compile_steps(steps: &[WorkflowStepSpec]) -> Result<Vec<Box<dyn WorkflowOp>>, MetalError> {
    let mut ops: Vec<Box<dyn WorkflowOp>> = Vec::with_capacity(steps.len());
    let registry = get_registry().read().unwrap();

    for step in steps {
        if let Some(builder) = registry.get(step.op.as_str()) {
            ops.push(builder(step.params.clone())?);
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
