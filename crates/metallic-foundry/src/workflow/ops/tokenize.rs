use crate::{
    error::MetalError, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct TokenizeOp {
    model_id: Option<String>,
    input_var: String,
    output_var: String,
    mode: Option<String>,
}

impl TokenizeOp {
    pub(crate) fn new(spec: crate::workflow::spec::TokenizeSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input_var: spec.input,
            output_var: spec.output,
            mode: spec.mode,
        }
    }
}

impl WorkflowOp for TokenizeOp {
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
        let tokens = match mode {
            "raw" => tokenizer.encode(text)?,
            "chat_single_turn" => tokenizer.encode_single_turn_chat_prompt(text)?,
            other => {
                return Err(MetalError::InvalidOperation(format!(
                    "TokenizeOp unsupported mode '{other}' (expected 'raw' or 'chat_single_turn')"
                )));
            }
        };

        if std::env::var("METALLIC_DEBUG_TOKENIZE").is_ok() || std::env::var("METALLIC_DEBUG_CHAT_TEMPLATE").is_ok() {
            let max_chars = 800usize;
            let shown = text.chars().take(max_chars).collect::<String>();
            let suffix = if text.chars().count() > max_chars { "…(truncated)" } else { "" };

            let head_n = 64usize.min(tokens.len());
            let token_head = &tokens[..head_n];
            let decoded_head = tokenizer
                .decode_lossless(token_head)
                .unwrap_or_else(|_| "<decode_error>".to_string());

            eprintln!(
                "[metallic][debug] TokenizeOp mode={mode} input='{}' chars={} tokens={} head_ids={:?}\n[metallic][debug] decoded_head:\n{}{}",
                self.input_var,
                text.chars().count(),
                tokens.len(),
                token_head,
                decoded_head,
                if tokens.len() > head_n {
                    "\n[metallic][debug] (decoded_head truncated to first 64 tokens)"
                } else {
                    ""
                }
            );
            eprintln!("[metallic][debug] input_text_head:\n{}{}", shown, suffix);
        }
        ctx.values.insert(self.output_var.clone(), Value::TokensU32(tokens));

        Ok(WorkflowOpOutcome::Continue)
    }
}
