use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct ComputeIntOp {
    output_var: String,
    expr: String,
}

impl ComputeIntOp {
    pub(crate) fn new(spec: crate::workflow::spec::ComputeIntSpec) -> Self {
        Self {
            output_var: spec.output,
            expr: spec.expr,
        }
    }
}

impl WorkflowOp for ComputeIntOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        // Simple expression evaluator for "{var} op literal" or "{var} op {var}".
        // Patterns:
        // - "{var} + 1"
        // - "{var} - 1"
        // - "{var} * {var}"
        // For now, let's just support basic tokenization and a simple stack or regex.

        // DEBT: This is a very crude implementation.
        let expr = self.expr.trim();

        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() == 1 {
            // Case: "output = {var}" or "output = 123"
            let val = self.resolve_term(parts[0], ctx)?;
            ctx.values.insert(self.output_var.clone(), Value::Usize(val));
            return Ok(WorkflowOpOutcome::Continue);
        }

        if parts.len() == 3 {
            let left = self.resolve_term(parts[0], ctx)?;
            let op = parts[1];
            let right = self.resolve_term(parts[2], ctx)?;

            let res = match op {
                "+" => left + right,
                "-" => left.saturating_sub(right),
                "*" => left * right,
                "/" => {
                    if right == 0 {
                        0
                    } else {
                        left / right
                    }
                }
                _ => return Err(MetalError::InvalidOperation(format!("Unsupported operator in expr: {}", op))),
            };

            ctx.values.insert(self.output_var.clone(), Value::Usize(res));
            return Ok(WorkflowOpOutcome::Continue);
        }

        Err(MetalError::InvalidOperation(format!(
            "Complex expressions not yet supported in ComputeInt: '{}'",
            self.expr
        )))
    }
}

impl ComputeIntOp {
    fn resolve_term(&self, term: &str, ctx: &WorkflowExecutionContext<'_>) -> Result<usize, MetalError> {
        if term.starts_with('{') && term.ends_with('}') {
            let var_name = &term[1..term.len() - 1];
            ctx.read_usize(var_name)
        } else if let Ok(v) = term.parse::<usize>() {
            Ok(v)
        } else {
            Err(MetalError::InvalidOperation(format!("Invalid term in expression: '{}'", term)))
        }
    }
}
