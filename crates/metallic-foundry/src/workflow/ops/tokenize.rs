use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct TokenizeOp {
    model_id: Option<String>,
    input_var: String,
    output_var: String,
    base_tokens_var: Option<String>,
    mode: Option<String>,
}

impl TokenizeOp {
    pub(crate) fn new(spec: crate::workflow::spec::TokenizeSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input_var: spec.input,
            output_var: spec.output,
            base_tokens_var: spec.base_tokens,
            mode: spec.mode,
        }
    }
}

impl WorkflowOp for TokenizeOp {
    fn memoize_spec(&self) -> Option<crate::workflow::ops::MemoizeSpec> {
        Some(crate::workflow::ops::MemoizeSpec {
            inputs: vec![self.input_var.clone()],
            outputs: vec![self.output_var.clone()],
            disable_in_tui: false,
        })
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;
        let tokenizer = model.tokenizer()?;

        // Resolve input text.
        let text = if let Some(val) = ctx.values.get(&self.input_var) {
            val.as_text()
                .ok_or_else(|| MetalError::InvalidOperation(format!("TokenizeOp input variable '{}' is not text", self.input_var)))?
        } else {
            // Assume the input_var IS the literal text if not found in variables?
            // Actually, the spec should probably distinguish between literals and variables.
            // For now, let's just assume it's a variable.
            return Err(MetalError::InvalidOperation(format!(
                "TokenizeOp missing input variable '{}'",
                self.input_var
            )));
        };

        let mode = self.mode.as_deref().unwrap_or("raw");
        let tokens_delta = match mode {
            "raw" => tokenizer.encode(text)?,
            "chat_single_turn" => tokenizer.encode_single_turn_chat_prompt(text)?,
            "delta" => tokenizer.encode(text)?,
            other => {
                return Err(MetalError::InvalidOperation(format!(
                    "TokenizeOp unsupported mode '{other}' (expected 'raw', 'chat_single_turn', or 'delta')"
                )));
            }
        };

        let tokens = if mode == "delta" {
            let base_name = self
                .base_tokens_var
                .as_deref()
                .ok_or_else(|| MetalError::InvalidOperation("TokenizeOp(mode=delta) requires 'base_tokens'".into()))?;
            let base = ctx
                .values
                .get(base_name)
                .and_then(|v| v.as_tokens_u32())
                .ok_or_else(|| MetalError::InvalidOperation(format!("TokenizeOp missing base_tokens '{base_name}' (u32[])")))?;
            // Build full prompt tokens by appending delta.
            let mut full: Vec<u32> = Vec::with_capacity(base.len().saturating_add(tokens_delta.len()));
            full.extend_from_slice(base);
            full.extend_from_slice(&tokens_delta);
            full
        } else {
            tokens_delta
        };

        if metallic_instrumentation::logging::debug_tokenize_or_template_enabled() {
            let head_n = 64usize.min(tokens.len());
            let token_head = &tokens[..head_n];
            let decoded_head = tokenizer
                .decode_lossless(token_head)
                .unwrap_or_else(|_| "<decode_error>".to_string());
            metallic_instrumentation::logging::emit_tokenize_debug_snapshot(&self.input_var, mode, text, &tokens, &decoded_head);
        }
        ctx.values.insert(self.output_var.clone(), Value::TokensU32(tokens));

        Ok(WorkflowOpOutcome::Continue)
    }
}
