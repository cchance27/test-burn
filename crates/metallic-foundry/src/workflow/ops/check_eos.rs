use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

fn ignore_eos_stop() -> bool {
    static IGNORE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IGNORE.get_or_init(|| std::env::var("METALLIC_IGNORE_EOS_STOP").is_ok_and(|v| v != "0"))
}

pub(crate) struct CheckEosOp {
    input: String,
    output: String,
    eos_token: Param<u32>,
}

impl CheckEosOp {
    pub(crate) fn new(spec: crate::workflow::spec::CheckEosSpec) -> Self {
        Self {
            input: spec.input,
            output: spec.output,
            eos_token: spec.eos_token,
        }
    }
}

impl WorkflowOp for CheckEosOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let token = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("CheckEosOp missing input token '{}' (u32)", self.input)))?;

        let eos = ctx.resolve_param_u32(&self.eos_token)?;

        let is_eos = !ignore_eos_stop() && token == eos;
        ctx.values.insert(self.output.clone(), Value::Bool(is_eos));

        Ok(WorkflowOpOutcome::Continue)
    }
}
