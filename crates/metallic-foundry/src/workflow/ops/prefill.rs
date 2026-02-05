use crate::{
    Foundry, error::MetalError, types::TensorArg, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext
    }
};

pub(crate) struct PrefillOp {
    model_id: Option<String>,
    input: String,
    output_pos: Option<String>,
    mode: Option<String>,
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
    pub(crate) fn new(spec: crate::workflow::spec::PrefillSpec) -> Self {
        Self {
            model_id: spec.model_id,
            input: spec.input,
            output_pos: spec.output_pos,
            mode: spec.mode,
            input_ids_binding: spec.input_ids_binding.unwrap_or_else(|| "input_ids".to_string()),
            logits_binding: spec.logits_binding.unwrap_or_else(|| "logits".to_string()),
            position_offset_key: spec.position_offset_key.unwrap_or_else(|| "position_offset".to_string()),
            m_key: spec.m_key.unwrap_or_else(|| "m".to_string()),
            seq_len_key: spec.seq_len_key.unwrap_or_else(|| "seq_len".to_string()),
            apply_derived_globals: spec.apply_derived_globals,
            description: spec.description,
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
        let prompt_tokens_full = ctx
            .values
            .get(&self.input)
            .and_then(|v| v.as_tokens_u32())
            .ok_or_else(|| MetalError::InvalidOperation(format!("Workflow prefill missing input '{}' (u32[])", self.input)))?;

        if prompt_tokens_full.is_empty() {
            return Err(MetalError::InvalidShape("prefill requires non-empty prompt_tokens".into()));
        }

        let conversation_id = conversation_id_from_ctx(ctx);
        let prefill_mode = self.mode.clone().unwrap_or_else(|| "delta".to_string());
        let kv_prefix_key = ctx
            .values
            .get("_internal.kv_prefix_key")
            .and_then(|v| v.as_text())
            .map(|v| v.to_string());
        let kv_prefix_base_key = ctx
            .values
            .get("_internal.kv_prefix_base_key")
            .and_then(|v| v.as_text())
            .map(|v| v.to_string());
        let setup_start = std::time::Instant::now();

        let (
            start_pos,
            prompt_len,
            end_pos,
            last_prefill_m,
            prefill_us,
            setup_us,
            logits_arg,
            token_source,
            execution_mode,
            effective_chunk_size,
            chunk_count,
            cache_hit_prefix_tokens,
            cache_lookup_path,
        ) =
            model.with_session_mut(ctx.foundry, |foundry: &mut Foundry, session| {
                let mode = prefill_mode.as_str();
                let original_start_pos = session.current_pos;
                let (prompt_tokens, token_source, cache_hit_prefix_tokens, cache_lookup_path): (
                    &[u32],
                    &'static str,
                    usize,
                    &'static str,
                ) = match mode {
                    "delta" => {
                        let mut start_pos = original_start_pos;
                        let mut prompt_tokens = prompt_tokens_full;
                        let mut token_source = "delta_input";
                        let mut cache_hit_prefix_tokens = 0usize;
                        let mut cache_lookup_path = "not_attempted";

                        if original_start_pos == 0 {
                            cache_lookup_path = "miss";

                            if let Some(key) = kv_prefix_key.as_deref()
                                && let Some(hit_prefix) =
                                    model.try_restore_kv_prefix_from_cache_key(foundry, session, key, prompt_tokens_full)?
                            {
                                apply_delta_cache_hit(
                                    prompt_tokens_full,
                                    &mut start_pos,
                                    &mut prompt_tokens,
                                    &mut token_source,
                                    &mut cache_hit_prefix_tokens,
                                    &mut cache_lookup_path,
                                    hit_prefix,
                                    "key_primary",
                                    "delta_cache_key_primary_suffix",
                                    "delta_cache_key_primary_full_replay_last",
                                );
                            }

                            if cache_hit_prefix_tokens == 0
                                && kv_prefix_base_key.as_deref() != kv_prefix_key.as_deref()
                                && let Some(key) = kv_prefix_base_key.as_deref()
                                && let Some(hit_prefix) =
                                    model.try_restore_kv_prefix_from_cache_key(foundry, session, key, prompt_tokens_full)?
                            {
                                apply_delta_cache_hit(
                                    prompt_tokens_full,
                                    &mut start_pos,
                                    &mut prompt_tokens,
                                    &mut token_source,
                                    &mut cache_hit_prefix_tokens,
                                    &mut cache_lookup_path,
                                    hit_prefix,
                                    "key_base",
                                    "delta_cache_key_base_suffix",
                                    "delta_cache_key_base_full_replay_last",
                                );
                            }

                            if cache_hit_prefix_tokens == 0
                                && let Some(hit_prefix) = model.try_restore_kv_prefix_from_cache(foundry, session, prompt_tokens_full)?
                            {
                                apply_delta_cache_hit(
                                    prompt_tokens_full,
                                    &mut start_pos,
                                    &mut prompt_tokens,
                                    &mut token_source,
                                    &mut cache_hit_prefix_tokens,
                                    &mut cache_lookup_path,
                                    hit_prefix,
                                    "token_prefix",
                                    "delta_cache_token_prefix_suffix",
                                    "delta_cache_token_prefix_full_replay_last",
                                );
                            }
                        }

                        session.current_pos = start_pos;
                        (prompt_tokens, token_source, cache_hit_prefix_tokens, cache_lookup_path)
                    }
                    "full_append_only" => {
                        if original_start_pos == 0 {
                            (prompt_tokens_full, "full_append_only_full", 0, "not_applicable")
                        } else if prompt_tokens_full.len() >= original_start_pos {
                            (
                                &prompt_tokens_full[original_start_pos..],
                                "full_append_only_suffix",
                                0,
                                "not_applicable",
                            )
                        } else {
                            return Err(MetalError::InvalidOperation(format!(
                                "prefill(mode=full_append_only) requires {}.len ({}) >= session.current_pos ({original_start_pos})",
                                self.input,
                                prompt_tokens_full.len()
                            )));
                        }
                    }
                    other => {
                        return Err(MetalError::InvalidOperation(format!(
                            "prefill unsupported mode '{other}' (expected 'delta' or 'full_append_only')"
                        )));
                    }
                };
                let start_pos = session.current_pos;
                let prompt_len = prompt_tokens.len();
                let max_new_tokens = ctx.values.get("max_tokens").and_then(|v| v.as_usize()).unwrap_or(0);
                let required_len = start_pos.saturating_add(prompt_len).saturating_add(max_new_tokens);
                tracing::debug!(
                    target: "metallic_foundry::workflow::ops::prefill",
                    conversation_id = conversation_id.as_str(),
                    mode,
                    token_source,
                    cache_hit_prefix_tokens,
                    cache_lookup_path,
                    kv_prefix_key = kv_prefix_key.as_deref().unwrap_or("<none>"),
                    kv_prefix_base_key = kv_prefix_base_key.as_deref().unwrap_or("<none>"),
                    prompt_tokens_full = prompt_tokens_full.len(),
                    prompt_tokens_selected = prompt_len,
                    start_pos,
                    max_new_tokens,
                    required_len,
                    "prefill prompt selection"
                );

                model.ensure_kv_capacity(
                    foundry,
                    &mut session.bindings,
                    &mut session.fast_bindings,
                    &mut session.context_config,
                    start_pos,
                    required_len,
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
                tracing::debug!(
                    target: "metallic_foundry::workflow::ops::prefill",
                    conversation_id = conversation_id.as_str(),
                    mode,
                    disable_batched_prefill,
                    profiling_per_kernel,
                    debug_sync,
                    max_prefill_chunk,
                    configured_prefill_chunk_size = prefill_chunk_size,
                    "prefill execution config"
                );

                let prefill_start = std::time::Instant::now();
                let setup_duration = prefill_start.duration_since(setup_start);
                let mut last_prefill_m = 1usize;

                let input_ids_key = self.input_ids_binding.as_str();
                let (execution_mode, effective_chunk_size, chunk_count) = if !disable_batched_prefill {
                    let execution_mode = "batched";
                    let chunk_size = {
                        let rebalance_chunk_size = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
                            let requested = requested.max(1).min(max_allowed.max(1));
                            if prompt_len <= 1 {
                                return 1;
                            }
                            let chunks = prompt_len.div_ceil(requested);
                            let balanced = prompt_len.div_ceil(chunks);
                            balanced.max(1).min(max_allowed)
                        };
                        rebalance_chunk_size(prompt_len, prefill_chunk_size, max_prefill_chunk)
                    };
                    let chunk_count = if prompt_len == 0 { 0 } else { prompt_len.div_ceil(chunk_size.max(1)) };

                    if !debug_sync && !profiling_per_kernel {
                        foundry.start_capture()?;
                    }

                    tracing::debug!(
                        target: "metallic_foundry::workflow::ops::prefill",
                        conversation_id = conversation_id.as_str(),
                        mode,
                        token_source,
                        execution_mode,
                        chunk_size,
                        chunk_count,
                        "prefill running batched chunk schedule"
                    );

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
                    (execution_mode, chunk_size, chunk_count)
                } else {
                    let execution_mode = "tokenwise";
                    let chunk_count = if prompt_len == 0 {
                        0
                    } else {
                        prompt_len.div_ceil(prefill_chunk_size.max(1))
                    };
                    tracing::debug!(
                        target: "metallic_foundry::workflow::ops::prefill",
                        conversation_id = conversation_id.as_str(),
                        mode,
                        token_source,
                        execution_mode,
                        chunk_size = prefill_chunk_size,
                        chunk_count,
                        "prefill running tokenwise chunk schedule"
                    );
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
                    (execution_mode, prefill_chunk_size, chunk_count)
                };

                let prefill_duration = prefill_start.elapsed();

                // Keep session state consistent for future KV growth/preservation and debugging.
                session.current_pos = start_pos + prompt_len;
                let end_pos = session.current_pos;

                if mode == "delta" && original_start_pos == 0 {
                    if let Err(err) =
                        model.store_kv_prefix_in_cache(foundry, session, prompt_tokens_full, kv_prefix_key.as_deref())
                    {
                        tracing::warn!(
                            target: "metallic_foundry::workflow::ops::prefill",
                            conversation_id = conversation_id.as_str(),
                            mode,
                            kv_prefix_key = kv_prefix_key.as_deref().unwrap_or("<none>"),
                            error = %err,
                            "prefill failed to store KV prefix snapshot"
                        );
                    }
                }

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
                let mut logits_arg = session.bindings.get(&self.logits_binding)?.clone();
                // The model spec allocates logits as [max_prefill_chunk, vocab_size]. After a batched prefill,
                // we must sample from the *last* token's logits (row = last_prefill_m - 1).
                //
                // If we sample from row 0, generation quality can collapse into highly repetitive loops.
                if logits_arg.dtype == crate::tensor::Dtype::F16 && logits_arg.dims.len() == 2 {
                    let rows = logits_arg.dims[0];
                    let vocab = logits_arg.dims[1];
                    if rows > 0 && vocab > 0 {
                        let last_row = last_prefill_m.saturating_sub(1).min(rows - 1);
                        let row_bytes = last_row.saturating_mul(vocab).saturating_mul(logits_arg.dtype.size_bytes());
                        logits_arg.offset = logits_arg.offset.saturating_add(row_bytes);
                        logits_arg.dims.clear();
                        logits_arg.dims.push(vocab);
                        logits_arg.strides.clear();
                        logits_arg.strides.push(1);
                    }
                }

                Ok((
                    start_pos,
                    prompt_len,
                    end_pos,
                    last_prefill_m,
                    prefill_duration.as_micros() as usize,
                    setup_duration.as_micros() as usize,
                    logits_arg,
                    token_source,
                    execution_mode,
                    effective_chunk_size,
                    chunk_count,
                    cache_hit_prefix_tokens,
                    cache_lookup_path,
                ))
            })?;

        tracing::debug!(
            target: "metallic_foundry::workflow::ops::prefill",
            conversation_id = conversation_id.as_str(),
            mode = prefill_mode.as_str(),
            token_source,
            execution_mode,
            cache_hit_prefix_tokens,
            cache_lookup_path,
            kv_prefix_key = kv_prefix_key.as_deref().unwrap_or("<none>"),
            kv_prefix_base_key = kv_prefix_base_key.as_deref().unwrap_or("<none>"),
            prompt_tokens_full = prompt_tokens_full.len(),
            prompt_len,
            start_pos,
            end_pos,
            last_prefill_m,
            chunk_size = effective_chunk_size,
            chunk_count,
            setup_us,
            prefill_us,
            "prefill completed"
        );

        ctx.values.insert("_internal.start_pos".to_string(), Value::Usize(start_pos));
        ctx.values.insert("_internal.prompt_len".to_string(), Value::Usize(prompt_len));
        ctx.values.insert("_internal.end_pos".to_string(), Value::Usize(end_pos));
        ctx.values
            .insert("_internal.last_prefill_m".to_string(), Value::Usize(last_prefill_m));
        ctx.values.insert("_internal.prefill_us".to_string(), Value::Usize(prefill_us));
        ctx.values.insert("_internal.setup_us".to_string(), Value::Usize(setup_us));
        ctx.values.insert(self.logits_binding.clone(), Value::Tensor(logits_arg));

        if let Some(out) = &self.output_pos {
            ctx.values.insert(out.clone(), Value::Usize(end_pos));
        }

        Ok(WorkflowOpOutcome::Continue)
    }
}

fn conversation_id_from_ctx(ctx: &WorkflowExecutionContext<'_>) -> String {
    let Some(value) = ctx.values.get("conversation_id") else {
        return "<default>".to_string();
    };
    if let Some(v) = value.as_text() {
        return v.to_string();
    }
    if let Some(v) = value.as_u32() {
        return v.to_string();
    }
    if let Some(v) = value.as_usize() {
        return v.to_string();
    }
    "<default>".to_string()
}

fn apply_delta_cache_hit<'a>(
    prompt_tokens_full: &'a [u32],
    start_pos: &mut usize,
    prompt_tokens: &mut &'a [u32],
    token_source: &mut &'static str,
    cache_hit_prefix_tokens: &mut usize,
    cache_lookup_path: &mut &'static str,
    hit_prefix: usize,
    lookup_path: &'static str,
    source_suffix: &'static str,
    source_full: &'static str,
) {
    *cache_hit_prefix_tokens = hit_prefix;
    *cache_lookup_path = lookup_path;

    if hit_prefix >= prompt_tokens_full.len() {
        if prompt_tokens_full.len() > 1 {
            *start_pos = prompt_tokens_full.len() - 1;
            *prompt_tokens = &prompt_tokens_full[*start_pos..];
            *token_source = source_full;
        } else {
            *start_pos = 0;
            *prompt_tokens = prompt_tokens_full;
            *token_source = "delta_cache_hit_single_fallback";
        }
    } else {
        *start_pos = hit_prefix;
        *prompt_tokens = &prompt_tokens_full[hit_prefix..];
        *token_source = source_suffix;
    }
}

#[cfg(test)]
mod tests {
    use super::apply_delta_cache_hit;

    #[test]
    fn partial_base_hit_uses_suffix_prefill() {
        let prompt_tokens_full = [10_u32, 11, 12, 13, 14, 15, 16, 17];
        let mut start_pos = 0usize;
        let mut prompt_tokens = &prompt_tokens_full[..];
        let mut token_source = "delta_input";
        let mut cache_hit_prefix_tokens = 0usize;
        let mut cache_lookup_path = "miss";

        apply_delta_cache_hit(
            &prompt_tokens_full,
            &mut start_pos,
            &mut prompt_tokens,
            &mut token_source,
            &mut cache_hit_prefix_tokens,
            &mut cache_lookup_path,
            5,
            "key_base",
            "delta_cache_key_base_suffix",
            "delta_cache_key_base_full_replay_last",
        );

        assert_eq!(start_pos, 5);
        assert_eq!(prompt_tokens, &prompt_tokens_full[5..]);
        assert_eq!(token_source, "delta_cache_key_base_suffix");
        assert_eq!(cache_hit_prefix_tokens, 5);
        assert_eq!(cache_lookup_path, "key_base");
    }
}
