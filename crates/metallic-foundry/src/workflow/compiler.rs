use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{LoopOp, PrefillOp, ReturnOp, SetGlobalsOp, SyncOp, WorkflowOp}
};
use crate::error::MetalError;

pub(crate) struct CompiledWorkflow {
    pub(crate) ops: Vec<Box<dyn WorkflowOp>>,
}

impl CompiledWorkflow {
    pub(crate) fn compile(workflow: &WorkflowSpec) -> Result<Self, MetalError> {
        let mut ops: Vec<Box<dyn WorkflowOp>> = Vec::with_capacity(workflow.steps.len());

        for step in &workflow.steps {
            match step {
                WorkflowStepSpec::Prefill {
                    model_id,
                    input,
                    input_ids_binding,
                    position_offset_key,
                    m_key,
                    seq_len_key,
                    apply_derived_globals,
                    description,
                } => {
                    ops.push(Box::new(PrefillOp::new(
                        model_id.clone(),
                        input.clone(),
                        input_ids_binding.clone(),
                        position_offset_key.clone(),
                        m_key.clone(),
                        seq_len_key.clone(),
                        *apply_derived_globals,
                        description.clone(),
                    )));
                }
                WorkflowStepSpec::Loop {
                    model_id,
                    condition,
                    args,
                    stages,
                } => {
                    ops.push(Box::new(LoopOp::new(
                        model_id.clone(),
                        condition.clone(),
                        args.clone(),
                        stages.clone(),
                    )?));
                }
                WorkflowStepSpec::SetGlobals {
                    model_id,
                    globals,
                    apply_derived_globals,
                } => {
                    ops.push(Box::new(SetGlobalsOp::new(
                        model_id.clone(),
                        globals.clone(),
                        *apply_derived_globals,
                    )));
                }
                WorkflowStepSpec::Synchronize => {
                    ops.push(Box::new(SyncOp));
                }
                WorkflowStepSpec::Return { output } => {
                    ops.push(Box::new(ReturnOp::new(output.clone())));
                }
            }
        }

        Ok(Self { ops })
    }
}
