use crate::{
    error::MetalError, metals::sampling::{ApplyRepetitionPenalty, RepetitionStateIngest, RepetitionStateInit, RepetitionStateUpdateFromToken, SampleTopK}, types::{MetalBuffer, MetalResourceOptions, TensorArg}, workflow::{
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
    // GPU-side repetition state (so penalties work without per-token CPU sorting/copy).
    rep_window_len: usize,
    rep_ring_buf: Option<MetalBuffer>,
    rep_ring_arg: Option<TensorArg>,
    rep_pairs_buf: Option<MetalBuffer>,
    rep_pairs_arg: Option<TensorArg>,
    rep_meta_buf: Option<MetalBuffer>,
    rep_meta_arg: Option<TensorArg>,
    rep_tokens_buf: Option<MetalBuffer>,
    rep_tokens_arg: Option<TensorArg>,
    rep_initialized: bool,
    rep_needs_prompt_ingest: bool,
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
            rep_window_len: 0,
            rep_ring_buf: None,
            rep_ring_arg: None,
            rep_pairs_buf: None,
            rep_pairs_arg: None,
            rep_meta_buf: None,
            rep_meta_arg: None,
            rep_tokens_buf: None,
            rep_tokens_arg: None,
            rep_initialized: false,
            rep_needs_prompt_ingest: true,
        }
    }
}

impl WorkflowOp for SampleOp {
    fn begin_run(&mut self, _ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        // Seed offset should restart per workflow invocation (per user turn).
        self.step = 0;
        self.rep_needs_prompt_ingest = true;
        Ok(())
    }

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

        // Apply token penalties in-place to logits before sampling (GPU-side state).
        let use_token_penalties = (repeat_penalty > 1.0 && repeat_penalty.is_finite())
            || (presence_penalty > 0.0 && presence_penalty.is_finite())
            || (frequency_penalty > 0.0 && frequency_penalty.is_finite());

        if use_token_penalties && repeat_last_n > 0 {
            let window_len = repeat_last_n.min(256);

            // (Re)allocate state buffers if needed.
            if self.rep_ring_buf.is_none() || self.rep_window_len != window_len {
                self.rep_window_len = window_len;
                self.rep_needs_prompt_ingest = true;
                self.rep_initialized = false;

                let ring_bytes = window_len * std::mem::size_of::<u32>();
                let pairs_bytes = window_len * 2 * std::mem::size_of::<u32>();
                let meta_bytes = 4 * std::mem::size_of::<u32>();

                let ring_buf = ctx
                    .foundry
                    .device
                    .new_buffer(ring_bytes.max(4), MetalResourceOptions::StorageModeShared)
                    .expect("Failed to allocate repetition ring buffer");
                let ring_arg = TensorArg::from_buffer(ring_buf.clone(), crate::tensor::Dtype::U32, vec![window_len], vec![1]);

                let pairs_buf = ctx
                    .foundry
                    .device
                    .new_buffer(pairs_bytes.max(8), MetalResourceOptions::StorageModeShared)
                    .expect("Failed to allocate repetition pairs buffer");
                let pairs_arg = TensorArg::from_buffer(pairs_buf.clone(), crate::tensor::Dtype::U32, vec![window_len * 2], vec![1]);

                let meta_buf = ctx
                    .foundry
                    .device
                    .new_buffer(meta_bytes, MetalResourceOptions::StorageModeShared)
                    .expect("Failed to allocate repetition meta buffer");
                let meta_arg = TensorArg::from_buffer(meta_buf.clone(), crate::tensor::Dtype::U32, vec![4], vec![1]);

                self.rep_ring_buf = Some(ring_buf);
                self.rep_ring_arg = Some(ring_arg);
                self.rep_pairs_buf = Some(pairs_buf);
                self.rep_pairs_arg = Some(pairs_arg);
                self.rep_meta_buf = Some(meta_buf);
                self.rep_meta_arg = Some(meta_arg);
            }

            // Ensure state is initialized/extended with the prompt tokens for this run.
            if self.rep_needs_prompt_ingest {
                let has_delta_prompt_tokens = ctx.values.contains_key("prompt_tokens_delta");
                let prompt_tokens = if has_delta_prompt_tokens {
                    ctx.values.get("prompt_tokens_delta").and_then(|v| v.as_tokens_u32())
                } else {
                    ctx.values.get("prompt_tokens").and_then(|v| v.as_tokens_u32())
                };

                let ring_arg = self.rep_ring_arg.as_ref().expect("rep_ring_arg set");
                let pairs_arg = self.rep_pairs_arg.as_ref().expect("rep_pairs_arg set");
                let meta_arg = self.rep_meta_arg.as_ref().expect("rep_meta_arg set");

                // Reset state for token-driven workflows (full prompt per run), but keep state for
                // message-driven workflows that provide delta prompt tokens across turns.
                if !self.rep_initialized || !has_delta_prompt_tokens {
                    let empty = ctx
                        .foundry
                        .device
                        .new_buffer(4, MetalResourceOptions::StorageModeShared)
                        .expect("Failed to allocate empty tokens buffer");
                    let empty_arg = TensorArg::from_buffer(empty, crate::tensor::Dtype::U32, vec![1], vec![1]);
                    let init = RepetitionStateInit::new(ring_arg, pairs_arg, meta_arg, &empty_arg, 0, window_len as u32);
                    ctx.foundry.run(&init)?;
                    self.rep_initialized = true;
                }

                if let Some(prompt_tokens) = prompt_tokens {
                    if !prompt_tokens.is_empty() {
                        let needed_bytes = prompt_tokens.len() * std::mem::size_of::<u32>();
                        let needs_alloc = self.rep_tokens_buf.as_ref().map(|b| b.length() < needed_bytes).unwrap_or(true);
                        if needs_alloc {
                            let new_buf = ctx
                                .foundry
                                .device
                                .new_buffer(needed_bytes.max(4), MetalResourceOptions::StorageModeShared)
                                .expect("Failed to allocate repetition tokens buffer");
                            self.rep_tokens_buf = Some(new_buf);
                        }
                        let buf = self.rep_tokens_buf.as_ref().expect("rep_tokens_buf set");
                        buf.copy_from_slice(prompt_tokens);
                        let tokens_arg = TensorArg::from_buffer(buf.clone(), crate::tensor::Dtype::U32, vec![prompt_tokens.len()], vec![1]);
                        self.rep_tokens_arg = Some(tokens_arg.clone());

                        let ingest = RepetitionStateIngest::new(
                            ring_arg,
                            pairs_arg,
                            meta_arg,
                            &tokens_arg,
                            prompt_tokens.len() as u32,
                            window_len as u32,
                        );
                        ctx.foundry.run(&ingest)?;
                    }
                }

                self.rep_needs_prompt_ingest = false;
            }

            let pairs_arg = self.rep_pairs_arg.as_ref().expect("rep_pairs_arg set");
            let penalty_kernel = ApplyRepetitionPenalty::new(
                logits_arg,
                pairs_arg,
                vocab_size,
                window_len as u32,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
            );
            ctx.foundry.run(&penalty_kernel)?;
        }

        let kernel = SampleTopK::new(logits_arg, out_arg, vocab_size, top_k, top_p, min_p, temp, seed);

        ctx.foundry.run(&kernel)?;

        // Update repetition state with the newly sampled token so the next step sees it.
        if use_token_penalties && repeat_last_n > 0 && self.rep_window_len > 0 {
            let ring_arg = self.rep_ring_arg.as_ref().expect("rep_ring_arg set");
            let pairs_arg = self.rep_pairs_arg.as_ref().expect("rep_pairs_arg set");
            let meta_arg = self.rep_meta_arg.as_ref().expect("rep_meta_arg set");
            let upd = RepetitionStateUpdateFromToken::new(ring_arg, pairs_arg, meta_arg, out_arg, self.rep_window_len as u32);
            ctx.foundry.run(&upd)?;
        }

        // Synchronize and read back
        let token = out_buffer.read_scalar::<u32>();
        ctx.values.insert(self.output_var.clone(), Value::U32(token));

        Ok(WorkflowOpOutcome::Continue)
    }
}
