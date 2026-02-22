use crate::{
    error::MetalError, workflow::{
        Value, channel::ChannelU32, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::{Param, StreamInitSpec, StreamWriteU32Spec}
    }
};

pub(crate) struct StreamInitOp {
    output: String,
    capacity: Param<u32>,
}

impl StreamInitOp {
    pub(crate) fn new(spec: StreamInitSpec) -> Self {
        Self {
            output: spec.output,
            capacity: spec.capacity,
        }
    }
}

impl WorkflowOp for StreamInitOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let cap = ctx.resolve_param_u32(&self.capacity)?;
        let chan = ChannelU32::allocate(ctx.foundry, cap)?;
        ctx.values.insert(self.output.clone(), Value::ChannelU32(chan));
        Ok(WorkflowOpOutcome::Continue)
    }
}

pub(crate) struct StreamWriteU32Op {
    channel: String,
    input: String,
}

impl StreamWriteU32Op {
    pub(crate) fn new(spec: StreamWriteU32Spec) -> Self {
        Self {
            channel: spec.channel,
            input: spec.input,
        }
    }
}

impl WorkflowOp for StreamWriteU32Op {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let chan = ctx
            .values
            .get(&self.channel)
            .and_then(|v| v.as_channel_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("stream_write_u32 missing channel '{}' (channel_u32)", self.channel)))?
            .clone();

        if let Some(t) = ctx.values.get(&self.input).and_then(|v| v.as_tensor()) {
            chan.push_value_buffer(ctx.foundry, t)?;
            return Ok(WorkflowOpOutcome::Continue);
        }

        let token = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("stream_write_u32 missing input '{}' (u32 or tensor)", self.input)))?;

        // Use scalar push kernel for v1.
        chan.push_scalar(ctx.foundry, token)?;
        Ok(WorkflowOpOutcome::Continue)
    }
}
