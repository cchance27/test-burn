use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::{CaptureBeginSpec, CaptureEndSpec, CaptureWaitSpec}
    }
};

pub(crate) struct CaptureBeginOp;

impl CaptureBeginOp {
    pub(crate) fn new(_spec: CaptureBeginSpec) -> Self {
        Self
    }
}

impl WorkflowOp for CaptureBeginOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        if !ctx.foundry.is_capturing() {
            ctx.foundry.start_capture()?;
        }
        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct CaptureEndOp {
    wait: bool,
    output: Option<String>,
}

impl CaptureEndOp {
    pub(crate) fn new(spec: CaptureEndSpec) -> Self {
        Self {
            wait: spec.wait,
            output: spec.output,
        }
    }
}

impl WorkflowOp for CaptureEndOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        if !ctx.foundry.is_capturing() {
            return Err(MetalError::InvalidOperation("capture_end requires an active capture".into()));
        }

        let cmd = ctx.foundry.end_capture()?;
        if self.wait {
            cmd.wait_until_completed();
        }

        if let Some(out) = &self.output {
            ctx.values.insert(out.clone(), Value::CommandBuffer(cmd));
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct CaptureWaitOp {
    input: String,
}

impl CaptureWaitOp {
    pub(crate) fn new(spec: CaptureWaitSpec) -> Self {
        Self { input: spec.input }
    }
}

impl WorkflowOp for CaptureWaitOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let cmd = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_command_buffer())
            .ok_or_else(|| MetalError::InvalidOperation(format!("capture_wait missing input '{}' (command_buffer)", self.input)))?;
        cmd.wait_until_completed();
        Ok(WorkflowOpOutcome::Continue)
    }
}
