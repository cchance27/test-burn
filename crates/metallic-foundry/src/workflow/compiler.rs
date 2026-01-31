use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{LoopOp, PrefillOp, ReturnOp, WorkflowOp}
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
                    description,
                } => {
                    ops.push(Box::new(PrefillOp::new(model_id.clone(), input.clone(), description.clone())));
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
                WorkflowStepSpec::Return { output } => {
                    ops.push(Box::new(ReturnOp::new(output.clone())));
                }
            }
        }

        Ok(Self { ops })
    }
}
