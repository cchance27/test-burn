use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{ComputeIntOp, DetokenizeOp, ForwardOp, LoopOp, PrefillOp, ReturnOp, SampleOp, SetGlobalsOp, SyncOp, TokenizeOp, WorkflowOp}
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
                WorkflowStepSpec::Forward {
                    model_id,
                    inputs,
                    outputs,
                    update_globals,
                    apply_derived_globals,
                    description,
                } => {
                    ops.push(Box::new(ForwardOp::new(
                        model_id.clone(),
                        inputs.clone(),
                        outputs.clone(),
                        update_globals.clone(),
                        *apply_derived_globals,
                        description.clone(),
                    )));
                }
                WorkflowStepSpec::Sample {
                    logits,
                    output,
                    temperature,
                    top_k,
                    top_p,
                    seed,
                } => {
                    ops.push(Box::new(SampleOp::new(
                        logits.clone(),
                        output.clone(),
                        temperature.clone(),
                        top_k.clone(),
                        top_p.clone(),
                        seed.clone(),
                    )));
                }
                WorkflowStepSpec::Tokenize { model_id, input, output } => {
                    ops.push(Box::new(TokenizeOp::new(model_id.clone(), input.clone(), output.clone())));
                }
                WorkflowStepSpec::Detokenize { model_id, input, output } => {
                    ops.push(Box::new(DetokenizeOp::new(model_id.clone(), input.clone(), output.clone())));
                }
                WorkflowStepSpec::ComputeInt { output, expr } => {
                    ops.push(Box::new(ComputeIntOp::new(output.clone(), expr.clone())));
                }
            }
        }

        Ok(Self { ops })
    }
}
