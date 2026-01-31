use crate::{
    error::MetalError, workflow::{
        ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct IfOp {
    condition: String,
    then_ops: Vec<Box<dyn WorkflowOp>>,
    else_ops: Vec<Box<dyn WorkflowOp>>,
}

impl IfOp {
    pub(crate) fn new(condition: String, then_ops: Vec<Box<dyn WorkflowOp>>, else_ops: Vec<Box<dyn WorkflowOp>>) -> Self {
        Self {
            condition,
            then_ops,
            else_ops,
        }
    }
}

impl WorkflowOp for IfOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let cond_val = ctx
            .values
            .get(&self.condition)
            .ok_or_else(|| MetalError::InvalidOperation(format!("IfOp missing condition variable '{}'", self.condition)))?;

        let should_run_then = match cond_val {
            crate::workflow::Value::Bool(b) => *b,
            crate::workflow::Value::U32(v) => *v != 0,
            crate::workflow::Value::Usize(v) => *v != 0,
            _ => {
                return Err(MetalError::InvalidOperation(format!(
                    "IfOp condition '{}' is not a boolean or integer",
                    self.condition
                )));
            }
        };

        let ops = if should_run_then { &mut self.then_ops } else { &mut self.else_ops };

        for op in ops {
            match op.execute(ctx, on_token)? {
                WorkflowOpOutcome::Continue => {}
                outcome => return Ok(outcome),
            }
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct WhileOp {
    condition: String,
    max_iterations: Option<Param<usize>>,
    body_ops: Vec<Box<dyn WorkflowOp>>,
}

impl WhileOp {
    pub(crate) fn new(condition: String, max_iterations: Option<Param<usize>>, body_ops: Vec<Box<dyn WorkflowOp>>) -> Self {
        Self {
            condition,
            max_iterations,
            body_ops,
        }
    }
}

impl WorkflowOp for WhileOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let max_iters = if let Some(p) = &self.max_iterations {
            Some(ctx.resolve_param_usize(p)?)
        } else {
            None
        };

        let mut iter = 0;
        loop {
            if let Some(max) = max_iters {
                if iter >= max {
                    break;
                }
            }

            // If the condition is a variable name, we lookup the variable.
            let cond_val = ctx
                .values
                .get(&self.condition)
                .ok_or_else(|| MetalError::InvalidOperation(format!("WhileOp missing condition variable '{}'", self.condition)))?;

            let should_run = match cond_val {
                crate::workflow::Value::Bool(b) => *b,
                crate::workflow::Value::U32(v) => *v != 0,
                crate::workflow::Value::Usize(v) => *v != 0,
                _ => {
                    return Err(MetalError::InvalidOperation(format!(
                        "WhileOp condition '{}' is not a boolean or integer",
                        self.condition
                    )));
                }
            };

            if !should_run {
                break;
            }

            let mut break_loop = false;
            for op in &mut self.body_ops {
                match op.execute(ctx, on_token)? {
                    WorkflowOpOutcome::Continue => {}
                    WorkflowOpOutcome::Return => return Ok(WorkflowOpOutcome::Return),
                    WorkflowOpOutcome::Break => {
                        break_loop = true;
                        break;
                    }
                    WorkflowOpOutcome::LoopContinue => {
                        // Stop executing body, proceed to next iteration
                        break;
                    }
                }
            }

            if break_loop {
                break;
            }

            iter += 1;
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct BreakOp;

impl WorkflowOp for BreakOp {
    fn execute(
        &mut self,
        _ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        Ok(WorkflowOpOutcome::Break)
    }
}

pub(crate) struct ContinueOp;

impl WorkflowOp for ContinueOp {
    fn execute(
        &mut self,
        _ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        Ok(WorkflowOpOutcome::LoopContinue)
    }
}
