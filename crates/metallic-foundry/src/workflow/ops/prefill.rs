use crate::{
    Foundry, error::MetalError, types::TensorArg, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct PrefillOp {
    model_id: Option<String>,
    input: String,
    input_ids_binding: String,
    logits_binding: String,
    position_offset_key: String,
    m_key: String,
    seq_len_key: String,
    apply_derived_globals: bool,
    #[allow(dead_code)]
    description: Option<String>,
}

impl PrefillOp {
    pub(crate) fn new(
        model_id: Option<String>,
        input: String,
        input_ids_binding: Option<String>,
        logits_binding: Option<String>,
        position_offset_key: Option<String>,
        m_key: Option<String>,
        seq_len_key: Option<String>,
        apply_derived_globals: bool,
        description: Option<String>,
    ) -> Self {
        Self {
            model_id,
            input,
            input_ids_binding: input_ids_binding.unwrap_or_else(|| "input_ids".to_string()),
            logits_binding: logits_binding.unwrap_or_else(|| "logits".to_string()),
            position_offset_key: position_offset_key.unwrap_or_else(|| "position_offset".to_string()),
            m_key: m_key.unwrap_or_else(|| "m".to_string()),
            seq_len_key: seq_len_key.unwrap_or_else(|| "seq_len".to_string()),
            apply_derived_globals,
            description,
        }
    }

    fn prefill_config() -> (usize, usize) {
        const DEFAULT_MAX_PREFILL_CHUNK: usize = 32;
        const DEFAULT_PREFILL_CHUNK_SIZE: usize = 32;
        const MAX_ALLOWED: usize = 512;

        let read = |var: &str| -> Option<usize> { std::env::var(var).ok().and_then(|v| v.trim().parse::<usize>().ok()) };
        let mut max_prefill_chunk = read("METALLIC_MAX_PREFILL_CHUNK").unwrap_or(DEFAULT_MAX_PREFILL_CHUNK);
        let mut prefill_chunk_size = read("METALLIC_PREFILL_CHUNK_SIZE").unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);

        max_prefill_chunk = max_prefill_chunk.clamp(1, MAX_ALLOWED);
        prefill_chunk_size = prefill_chunk_size.clamp(1, MAX_ALLOWED);

        if prefill_chunk_size > max_prefill_chunk {
            max_prefill_chunk = prefill_chunk_size;
        }

        (max_prefill_chunk, prefill_chunk_size)
    }
}

impl WorkflowOp for PrefillOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;
        let prompt_tokens = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_tokens_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("Workflow prefill missing input '{}' (u32[])", self.input)))?;

        if prompt_tokens.is_empty() {
            return Err(MetalError::InvalidShape("prefill requires non-empty prompt_tokens".into()));
        }

        let setup_start = std::time::Instant::now();

        let (start_pos, prompt_len, last_prefill_m, prefill_us, setup_us, logits_arg) =
            model.with_session_mut(ctx.foundry, |foundry: &mut Foundry, session| {
                let start_pos = session.current_pos;
                let prompt_len = prompt_tokens.len();

                model.ensure_kv_capacity(
                    foundry,
                    &mut session.bindings,
                    &mut session.fast_bindings,
                    &mut session.context_config,
                    start_pos,
                    start_pos + prompt_len,
                )?;

                if prompt_len + start_pos > session.context_config.max_context_len {
                    return Err(MetalError::InvalidShape(format!(
                        "Prompt length {prompt_len} + start_pos {start_pos} exceeds max_context_len {}",
                        session.context_config.max_context_len
                    )));
                }

                let input_ids_full_arg = session.bindings.get("input_ids_full")?;
                let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap();
                input_ids_full.copy_from_slice_offset(prompt_tokens, start_pos);

                // Defaults for decode (m=1, seq_len=1). Prefill overrides these per chunk.
                model.set_int_global(&mut session.bindings, &self.m_key, 1);
                model.set_int_global(&mut session.bindings, &self.seq_len_key, 1);
                model.set_int_global(&mut session.bindings, &self.position_offset_key, start_pos);
                if self.apply_derived_globals {
                    model.apply_derived_globals(&mut session.bindings);
                }

                let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
                let disable_batched_prefill_env = std::env::var("METALLIC_DISABLE_BATCHED_PREFILL").is_ok();
                let disable_batched_prefill = profiling_per_kernel || disable_batched_prefill_env;
                let debug_sync = std::env::var("METALLIC_DEBUG_FORWARD_SYNC").is_ok();

                let max_prefill_chunk = session.bindings.get_int_global("max_prefill_chunk").unwrap_or(32).max(1);
                let (_, mut prefill_chunk_size) = Self::prefill_config();
                prefill_chunk_size = prefill_chunk_size.min(max_prefill_chunk).max(1);

                let prefill_start = std::time::Instant::now();
                let setup_duration = prefill_start.duration_since(setup_start);
                let mut last_prefill_m = 1usize;

                let input_ids_key = self.input_ids_binding.as_str();

                if !disable_batched_prefill {
                    if !debug_sync && !profiling_per_kernel {
                        foundry.start_capture()?;
                    }

                    let rebalance_chunk_size = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
                        let requested = requested.max(1).min(max_allowed.max(1));
                        if prompt_len <= 1 {
                            return 1;
                        }
                        let chunks = prompt_len.div_ceil(requested);
                        let balanced = prompt_len.div_ceil(chunks);
                        balanced.max(1).min(max_allowed)
                    };

                    let chunk_size = rebalance_chunk_size(prompt_len, prefill_chunk_size, max_prefill_chunk);

                    for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                        let m = chunk_tokens.len();
                        last_prefill_m = m;
                        let base_pos = start_pos + chunk_idx * chunk_size;

                        if debug_sync && !profiling_per_kernel {
                            foundry.start_capture()?;
                        }

                        model.set_int_global(&mut session.bindings, &self.m_key, m);
                        model.set_int_global(&mut session.bindings, &self.seq_len_key, m);
                        model.set_int_global(&mut session.bindings, &self.position_offset_key, base_pos);
                        if self.apply_derived_globals {
                            model.apply_derived_globals(&mut session.bindings);
                        }

                        let input_ids_full_arg = session.bindings.get("input_ids_full")?;
                        let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap().clone();
                        let mut tensor_input = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![m], vec![1]);
                        tensor_input.offset = base_pos * 4;
                        model.set_binding(&mut session.bindings, &mut session.fast_bindings, input_ids_key, tensor_input);

                        model.forward(foundry, &mut session.bindings, &session.fast_bindings)?;

                        if debug_sync && !profiling_per_kernel {
                            let cmd = foundry.end_capture()?;
                            cmd.wait_until_completed();
                        }
                    }

                    if !debug_sync && !profiling_per_kernel {
                        let cmd = foundry.end_capture()?;
                        cmd.wait_until_completed();
                    }
                } else {
                    model.set_int_global(&mut session.bindings, &self.m_key, 1);
                    model.set_int_global(&mut session.bindings, &self.seq_len_key, 1);

                    for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(prefill_chunk_size).enumerate() {
                        let base_pos = start_pos + chunk_idx * prefill_chunk_size;

                        if !profiling_per_kernel && (debug_sync || chunk_idx == 0) {
                            foundry.start_capture()?;
                        }

                        for i in 0..chunk_tokens.len() {
                            let pos = base_pos + i;
                            model.set_int_global(&mut session.bindings, &self.position_offset_key, pos);
                            if self.apply_derived_globals {
                                model.apply_derived_globals(&mut session.bindings);
                            }

                            let input_ids_full_arg = session.bindings.get("input_ids_full")?;
                            let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap().clone();
                            let mut tensor_input = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![1], vec![1]);
                            tensor_input.offset = pos * 4;
                            model.set_binding(&mut session.bindings, &mut session.fast_bindings, input_ids_key, tensor_input);

                            model.forward(foundry, &mut session.bindings, &session.fast_bindings)?;
                        }

                        if debug_sync && !profiling_per_kernel {
                            let cmd = foundry.end_capture()?;
                            cmd.wait_until_completed();
                        }
                    }

                    if !debug_sync && !profiling_per_kernel {
                        let cmd = foundry.end_capture()?;
                        cmd.wait_until_completed();
                    }
                }

                let prefill_duration = prefill_start.elapsed();

                // Reset to decode mode for autoregressive decode (M=1).
                model.set_int_global(&mut session.bindings, &self.m_key, 1);
                model.set_int_global(&mut session.bindings, &self.seq_len_key, 1);
                model.set_int_global(&mut session.bindings, &self.position_offset_key, start_pos + prompt_len);
                if self.apply_derived_globals {
                    model.apply_derived_globals(&mut session.bindings);
                }

                // Ensure input_ids is bound to any valid U32 buffer; decode stage overwrites it to sampled-token buffers.
                {
                    let input_ids_full_arg = session.bindings.get("input_ids_full")?;
                    let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap().clone();
                    let mut tensor_input = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![1], vec![1]);
                    tensor_input.offset = 0;
                    model.set_binding(&mut session.bindings, &mut session.fast_bindings, input_ids_key, tensor_input);
                }

                // Extract logits for use in subsequent ops.
                let logits_arg = session.bindings.get(&self.logits_binding)?.clone();

                Ok((
                    start_pos,
                    prompt_len,
                    last_prefill_m,
                    prefill_duration.as_micros() as usize,
                    setup_duration.as_micros() as usize,
                    logits_arg,
                ))
            })?;

        ctx.values.insert("_internal.start_pos".to_string(), Value::Usize(start_pos));
        ctx.values.insert("_internal.prompt_len".to_string(), Value::Usize(prompt_len));
        ctx.values
            .insert("_internal.last_prefill_m".to_string(), Value::Usize(last_prefill_m));
        ctx.values.insert("_internal.prefill_us".to_string(), Value::Usize(prefill_us));
        ctx.values.insert("_internal.setup_us".to_string(), Value::Usize(setup_us));
        ctx.values.insert(self.logits_binding.clone(), Value::Tensor(logits_arg));

        Ok(WorkflowOpOutcome::Continue)
    }
}
