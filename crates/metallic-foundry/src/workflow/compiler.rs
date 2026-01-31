use super::{
    WorkflowSpec, WorkflowStepSpec, ops::{
        AppendTokenOp, BreakOp, CheckEosOp, ComputeIntOp, ContinueOp, DetokenizeOp, ForwardOp, GraphForwardOp, IfOp, PrefillOp, ReturnOp, SampleOp, SetGlobalsOp, SyncOp, TokenizeOp, WhileOp, WorkflowOp
    }
};
use crate::error::MetalError;

pub(crate) struct CompiledWorkflow {
    pub(crate) ops: Vec<Box<dyn WorkflowOp>>,
}

fn compile_steps(steps: &[WorkflowStepSpec]) -> Result<Vec<Box<dyn WorkflowOp>>, MetalError> {
    let mut ops: Vec<Box<dyn WorkflowOp>> = Vec::with_capacity(steps.len());

    for step in steps {
        match step {
            WorkflowStepSpec::Prefill {
                model_id,
                input,
                input_ids_binding,
                logits_binding,
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
                    logits_binding.clone(),
                    position_offset_key.clone(),
                    m_key.clone(),
                    seq_len_key.clone(),
                    *apply_derived_globals,
                    description.clone(),
                )));
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
            WorkflowStepSpec::If { condition, then, else_ } => {
                ops.push(Box::new(IfOp::new(condition.clone(), compile_steps(then)?, compile_steps(else_)?)));
            }
            WorkflowStepSpec::While {
                condition,
                max_iterations,
                body,
            } => {
                ops.push(Box::new(WhileOp::new(
                    condition.clone(),
                    max_iterations.clone(),
                    compile_steps(body)?,
                )));
            }
            WorkflowStepSpec::Break => {
                ops.push(Box::new(BreakOp));
            }
            WorkflowStepSpec::Continue => {
                ops.push(Box::new(ContinueOp));
            }
            WorkflowStepSpec::CheckEos { input, output, eos_token } => {
                ops.push(Box::new(CheckEosOp::new(input.clone(), output.clone(), eos_token.clone())));
            }
            WorkflowStepSpec::AppendToken { input, output } => {
                ops.push(Box::new(AppendTokenOp::new(input.clone(), output.clone())));
            }
            WorkflowStepSpec::GraphForward {
                model_id,
                token_var,
                input_ids_binding,
                logits_binding,
                position_offset_key,
                position,
                apply_derived_globals,
                description,
            } => {
                ops.push(Box::new(GraphForwardOp::new(
                    model_id.clone(),
                    token_var.clone(),
                    input_ids_binding.clone(),
                    logits_binding.clone(),
                    position_offset_key.clone(),
                    position.clone(),
                    *apply_derived_globals,
                    description.clone(),
                )));
            }
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
