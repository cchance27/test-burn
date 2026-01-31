use std::collections::BTreeMap;

use crate::{
    error::MetalError, workflow::{
        ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct SetGlobalsOp {
    model_id: Option<String>,
    globals: BTreeMap<String, Param<usize>>,
    apply_derived_globals: bool,
}

impl SetGlobalsOp {
    pub(crate) fn new(model_id: Option<String>, globals: BTreeMap<String, Param<usize>>, apply_derived_globals: bool) -> Self {
        Self {
            model_id,
            globals,
            apply_derived_globals,
        }
    }
}

impl WorkflowOp for SetGlobalsOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;

        let mut resolved: Vec<(&str, usize)> = Vec::with_capacity(self.globals.len());
        for (key, value) in &self.globals {
            resolved.push((key.as_str(), ctx.resolve_param_usize(value)?));
        }

        model.with_session_mut(ctx.foundry, |_foundry, session| {
            for (key, v) in &resolved {
                model.set_global_usize(&mut session.bindings, *key, *v);
            }
            if self.apply_derived_globals {
                model.apply_derived_globals(&mut session.bindings);
            }
            Ok(())
        })?;

        Ok(WorkflowOpOutcome::Continue)
    }
}
