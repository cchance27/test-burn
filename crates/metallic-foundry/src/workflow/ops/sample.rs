use crate::{
    error::MetalError, metals::sampling::{ApplyRepetitionPenalty, SampleTopK}, types::{MetalBuffer, MetalResourceOptions, TensorArg}, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::Param
    }
};

pub(crate) struct SampleOp {
    logits_var: String,
    output_var: String,
    temperature: Param<f32>,
    top_k: Param<u32>,
    top_p: Param<f32>,
    min_p: Param<f32>,
    repeat_penalty: Param<f32>,
    repeat_last_n: Param<usize>,
    presence_penalty: Param<f32>,
    frequency_penalty: Param<f32>,
    seed: Param<u32>,
    step: u32,
    out_buffer: Option<MetalBuffer>,
    out_arg: Option<TensorArg>,
    recent_pairs_buf: Option<MetalBuffer>,
}

impl SampleOp {
    pub(crate) fn new(spec: crate::workflow::spec::SampleSpec) -> Self {
        Self {
            logits_var: spec.logits,
            output_var: spec.output,
            temperature: spec.temperature,
            top_k: spec.top_k,
            top_p: spec.top_p,
            min_p: spec.min_p,
            repeat_penalty: spec.repeat_penalty,
            repeat_last_n: spec.repeat_last_n,
            presence_penalty: spec.presence_penalty,
            frequency_penalty: spec.frequency_penalty,
            seed: spec.seed,
            step: 0,
            out_buffer: None,
            out_arg: None,
            recent_pairs_buf: None,
        }
    }
}

fn build_repetition_pairs(mut recent: Vec<u32>) -> Vec<u32> {
    if recent.is_empty() {
        return Vec::new();
    }

    recent.sort_unstable();

    // Packed (token_id, count) pairs.
    let mut pairs: Vec<u32> = Vec::with_capacity(recent.len().saturating_mul(2));
    let mut i = 0usize;
    while i < recent.len() {
        let tok = recent[i];
        i += 1;
        let mut count = 1u32;
        while i < recent.len() && recent[i] == tok {
            count = count.saturating_add(1);
            i += 1;
        }
        pairs.push(tok);
        pairs.push(count);
    }

    pairs
}

impl WorkflowOp for SampleOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let logits_arg = ctx
            .values
            .get(&self.logits_var)
            .and_then(|v| v.as_tensor())
            .ok_or_else(|| MetalError::InvalidOperation(format!("SampleOp missing logits variable '{}' (Tensor)", self.logits_var)))?;

        let temp = ctx.resolve_param_f32(&self.temperature)?;
        let top_k = ctx.resolve_param_u32(&self.top_k)?;
        let top_p = ctx.resolve_param_f32(&self.top_p)?;
        let min_p = ctx.resolve_param_f32(&self.min_p)?;
        let repeat_penalty = ctx.resolve_param_f32(&self.repeat_penalty)?;
        let repeat_last_n = ctx.resolve_param_usize(&self.repeat_last_n)?;
        let presence_penalty = ctx.resolve_param_f32(&self.presence_penalty)?;
        let frequency_penalty = ctx.resolve_param_f32(&self.frequency_penalty)?;
        let seed = ctx.resolve_param_u32(&self.seed)?.wrapping_add(self.step);

        self.step = self.step.wrapping_add(1);

        // Allocate and reuse a 1-token output buffer for the sampled token.
        if self.out_buffer.is_none() {
            let out_buffer = ctx
                .foundry
                .device
                .new_buffer(4, MetalResourceOptions::StorageModeShared)
                .expect("Failed to allocate sample output buffer");
            let out_arg = TensorArg::from_buffer(out_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            self.out_buffer = Some(out_buffer);
            self.out_arg = Some(out_arg);
        }
        let out_buffer = self.out_buffer.as_ref().expect("out_buffer set");
        let out_arg = self.out_arg.as_ref().expect("out_arg set");

        let vocab_size = logits_arg.dims()[logits_arg.dims().len() - 1] as u32;

        // Apply token penalties in-place to logits before sampling.
        let use_token_penalties = (repeat_penalty > 1.0 && repeat_penalty.is_finite())
            || (presence_penalty > 0.0 && presence_penalty.is_finite())
            || (frequency_penalty > 0.0 && frequency_penalty.is_finite());

        if use_token_penalties && repeat_last_n > 0 {
            let mut recent: Vec<u32> = Vec::with_capacity(repeat_last_n.min(256));

            if let Some(generated) = ctx.values.get("generated_tokens").and_then(|v| v.as_tokens_u32()) {
                let take = repeat_last_n.min(generated.len());
                let start = generated.len().saturating_sub(take);
                recent.extend_from_slice(&generated[start..]);
            }

            if recent.len() < repeat_last_n {
                if let Some(prompt_tokens) = ctx.values.get("prompt_tokens").and_then(|v| v.as_tokens_u32()) {
                    let need = repeat_last_n - recent.len();
                    let start = prompt_tokens.len().saturating_sub(need);
                    recent.extend_from_slice(&prompt_tokens[start..]);
                }
            }

            let pairs = build_repetition_pairs(recent);
            if !pairs.is_empty() {
                let needed_bytes = pairs.len() * std::mem::size_of::<u32>();
                let needs_alloc = self.recent_pairs_buf.as_ref().map(|b| b.length() < needed_bytes).unwrap_or(true);
                if needs_alloc {
                    let new_buf = ctx
                        .foundry
                        .device
                        .new_buffer(needed_bytes.max(4), MetalResourceOptions::StorageModeShared)
                        .expect("Failed to allocate recent_pairs buffer");
                    self.recent_pairs_buf = Some(new_buf);
                }
                let buf = self.recent_pairs_buf.as_ref().expect("recent_pairs_buf set");
                buf.copy_from_slice(&pairs);
                let recent_pairs_arg = TensorArg::from_buffer(buf.clone(), crate::tensor::Dtype::U32, vec![pairs.len()], vec![1]);
                let pair_len = (pairs.len() / 2) as u32;
                let penalty_kernel = ApplyRepetitionPenalty::new(
                    logits_arg,
                    &recent_pairs_arg,
                    vocab_size,
                    pair_len,
                    repeat_penalty,
                    presence_penalty,
                    frequency_penalty,
                );
                ctx.foundry.run(&penalty_kernel)?;
            }
        }

        let kernel = SampleTopK::new(logits_arg, out_arg, vocab_size, top_k, top_p, min_p, temp, seed);

        ctx.foundry.run(&kernel)?;

        // Synchronize and read back
        let token = out_buffer.read_scalar::<u32>();
        ctx.values.insert(self.output_var.clone(), Value::U32(token));

        Ok(WorkflowOpOutcome::Continue)
    }
}
