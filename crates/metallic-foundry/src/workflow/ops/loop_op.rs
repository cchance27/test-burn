use crate::{
    Foundry, error::MetalError, metals::sampling::SampleTopK, types::TensorArg, workflow::{
        Value, ops::{WorkflowOp, WorkflowOpOutcome}, runner::WorkflowExecutionContext, spec::{Param, WorkflowStageSpec}
    }
};

pub(crate) struct LoopOp {
    model_id: Option<String>,
    #[allow(dead_code)]
    condition: Option<String>,
    #[allow(dead_code)]
    args: Vec<String>,
    stages: Vec<WorkflowStageSpec>,
    append_output: String,
    sample_logits_binding: String,
    sample_token_var: String,
    input_ids_binding: String,
    position_offset_key: String,
    apply_derived_globals: bool,
    has_append_stage: bool,
}

#[derive(Clone)]
struct SampleStageCfg {
    temperature: Param<f32>,
    top_k: Param<u32>,
    top_p: Param<f32>,
    seed: Param<u32>,
}

#[derive(Clone)]
struct CheckEosStageCfg {
    eos_token: Param<u32>,
}

impl LoopOp {
    pub(crate) fn new(
        model_id: Option<String>,
        condition: Option<String>,
        args: Vec<String>,
        stages: Vec<WorkflowStageSpec>,
    ) -> Result<Self, MetalError> {
        if stages.is_empty() {
            return Err(MetalError::InvalidOperation("loop must contain at least one stage".into()));
        }

        let mut append_output: Option<String> = None;
        let mut has_append_stage = false;
        let mut sample_logits_binding: Option<String> = None;
        let mut sample_token_var: Option<String> = None;
        let mut has_check_eos = false;
        let mut input_ids_binding: Option<String> = None;
        let mut graph_forward_logits_binding: Option<String> = None;
        let mut graph_forward_token_var: Option<String> = None;
        let mut position_offset_key: Option<String> = None;
        let mut apply_derived_globals = true;
        let mut has_graph_forward = false;

        for stage in &stages {
            match stage {
                WorkflowStageSpec::Sample {
                    model_id: stage_model_id,
                    logits_binding,
                    output,
                    ..
                } => {
                    if sample_logits_binding.is_some() || sample_token_var.is_some() {
                        return Err(MetalError::InvalidOperation("loop must contain exactly one 'sample' stage".into()));
                    }
                    sample_logits_binding = Some(logits_binding.clone());
                    sample_token_var = Some(output.clone());
                    if let (Some(step_id), Some(stage_id)) = (model_id.as_deref(), stage_model_id.as_deref())
                        && step_id != stage_id
                    {
                        return Err(MetalError::InvalidOperation(format!(
                            "loop stage model_id '{stage_id}' does not match loop model_id '{step_id}'"
                        )));
                    }
                }
                WorkflowStageSpec::CheckEos { .. } => {
                    has_check_eos = true;
                }
                WorkflowStageSpec::AppendToken { output, .. } => {
                    append_output = Some(output.clone());
                    has_append_stage = true;
                }
                WorkflowStageSpec::GraphForward {
                    model_id: stage_model_id,
                    token_var,
                    input_ids_binding: stage_input_ids_binding,
                    logits_binding,
                    position_offset_key: stage_position_offset_key,
                    apply_derived_globals: stage_apply_derived_globals,
                    ..
                } => {
                    has_graph_forward = true;
                    graph_forward_logits_binding = Some(logits_binding.clone());
                    graph_forward_token_var = Some(token_var.clone());
                    input_ids_binding = Some(stage_input_ids_binding.clone().unwrap_or_else(|| "input_ids".to_string()));
                    position_offset_key = Some(stage_position_offset_key.clone().unwrap_or_else(|| "position_offset".to_string()));
                    apply_derived_globals = *stage_apply_derived_globals;
                    if let (Some(step_id), Some(stage_id)) = (model_id.as_deref(), stage_model_id.as_deref())
                        && step_id != stage_id
                    {
                        return Err(MetalError::InvalidOperation(format!(
                            "loop stage model_id '{stage_id}' does not match loop model_id '{step_id}'"
                        )));
                    }
                }
            }
        }

        let sample_logits_binding =
            sample_logits_binding.ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'sample'".into()))?;
        let sample_token_var =
            sample_token_var.ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'sample'".into()))?;

        if !has_check_eos {
            return Err(MetalError::InvalidOperation("loop missing required stage 'check_eos'".into()));
        }
        if !has_graph_forward {
            return Err(MetalError::InvalidOperation("loop missing required stage 'graph_forward'".into()));
        }
        let append_output = append_output.unwrap_or_else(|| "generated_tokens".to_string());

        let input_ids_binding = input_ids_binding.unwrap_or_else(|| "input_ids".to_string());
        let position_offset_key = position_offset_key.unwrap_or_else(|| "position_offset".to_string());
        let graph_forward_logits_binding = graph_forward_logits_binding
            .ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'graph_forward'".into()))?;
        let graph_forward_token_var =
            graph_forward_token_var.ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'graph_forward'".into()))?;

        if graph_forward_token_var != sample_token_var {
            return Err(MetalError::InvalidOperation(format!(
                "graph_forward.token_var '{graph_forward_token_var}' must match sample.output '{sample_token_var}'"
            )));
        }
        if graph_forward_logits_binding != sample_logits_binding {
            return Err(MetalError::InvalidOperation(format!(
                "graph_forward.logits_binding '{graph_forward_logits_binding}' must match sample.logits_binding '{sample_logits_binding}'"
            )));
        }

        Ok(Self {
            model_id,
            condition,
            args,
            stages,
            append_output,
            sample_logits_binding,
            sample_token_var,
            input_ids_binding,
            position_offset_key,
            apply_derived_globals,
            has_append_stage,
        })
    }

    fn decode_batch_size(bindings: &crate::spec::TensorBindings) -> usize {
        let default_decode_batch_size = bindings
            .get("output_weight")
            .ok()
            .map(|w| if w.dtype == crate::tensor::Dtype::F16 { 16 } else { 64 })
            .unwrap_or(64);

        const MAX: usize = 256;
        let parsed = std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        parsed.unwrap_or(default_decode_batch_size).clamp(1, MAX)
    }

    fn ignore_eos_stop_enabled() -> bool {
        const VAR: &str = "METALLIC_IGNORE_EOS_STOP";
        let Ok(value) = std::env::var(VAR) else {
            return false;
        };
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return false;
        }
        let lowered = trimmed.to_ascii_lowercase();
        !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
    }
}

impl WorkflowOp for LoopOp {
    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let model = ctx.resolve_model(self.model_id.as_deref())?;

        let max_tokens_key = self.args.get(0).map(|s| s.as_str()).unwrap_or("max_tokens");
        let max_new_tokens = ctx.read_usize(max_tokens_key)?;

        let start_pos = ctx.read_usize("_internal.start_pos")?;
        let prompt_len = ctx.read_usize("_internal.prompt_len")?;
        let last_prefill_m = ctx.read_usize("_internal.last_prefill_m")?;
        let prefill_duration = std::time::Duration::from_micros(ctx.read_usize("_internal.prefill_us")? as u64);
        let setup_duration = std::time::Duration::from_micros(ctx.read_usize("_internal.setup_us")? as u64);

        let mut sample_cfg: Option<SampleStageCfg> = None;
        let mut eos_cfg: Option<CheckEosStageCfg> = None;
        for stage in &self.stages {
            match stage {
                WorkflowStageSpec::Sample {
                    temperature,
                    top_k,
                    top_p,
                    seed,
                    ..
                } => {
                    sample_cfg = Some(SampleStageCfg {
                        temperature: temperature.clone(),
                        top_k: top_k.clone(),
                        top_p: top_p.clone(),
                        seed: seed.clone(),
                    });
                }
                WorkflowStageSpec::CheckEos { eos_token, .. } => {
                    eos_cfg = Some(CheckEosStageCfg {
                        eos_token: eos_token.clone(),
                    });
                }
                _ => {}
            }
        }

        let sample_cfg = sample_cfg.ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'sample'".into()))?;
        let eos_cfg = eos_cfg.ok_or_else(|| MetalError::InvalidOperation("loop missing required stage 'check_eos'".into()))?;

        let temperature = ctx.resolve_param_f32(&sample_cfg.temperature)?;
        let top_k = ctx.resolve_param_u32(&sample_cfg.top_k)?;
        let top_p = ctx.resolve_param_f32(&sample_cfg.top_p)?;
        let seed = ctx.resolve_param_u32(&sample_cfg.seed)?;
        let eos_token = ctx.resolve_param_u32(&eos_cfg.eos_token)?;

        let generated = model.with_session_mut(ctx.foundry, |foundry: &mut Foundry, session| {
            let bindings = &mut session.bindings;
            let fast_bindings = &mut session.fast_bindings;

            // Ensure input_ids is bound to any valid U32 buffer; loop overwrites it to sampled-token buffers.
            {
                let input_ids_full_arg = bindings.get("input_ids_full")?;
                let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap().clone();
                let mut tensor_input = TensorArg::from_buffer(
                    input_ids_full,
                    crate::tensor::Dtype::U32,
                    vec![1],
                    vec![1],
                );
                tensor_input.offset = 0;
                model.set_binding(bindings, fast_bindings, &self.input_ids_binding, tensor_input);
            }

            let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
            let emit_host_metrics = crate::instrument::foundry_metrics_enabled();

            let batch_size = if profiling_per_kernel { 1 } else { Self::decode_batch_size(bindings) };
            let sample_out_count = bindings.iter().filter(|(k, _)| k.starts_with("sample_out_")).count();
            if batch_size > sample_out_count {
                return Err(MetalError::InvalidShape(format!(
                    "Decode batch_size {batch_size} exceeds session capacity {sample_out_count}. This typically means METALLIC_FOUNDRY_DECODE_BATCH_SIZE changed after model load."
                )));
            }

            let ignore_eos_stop = Self::ignore_eos_stop_enabled();
            let greedy = temperature <= 0.0 || !temperature.is_finite() || top_k == 0;
            let vocab_size = model.architecture().vocab_size() as u32;

            let mut pending_count = 0usize;
            let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
            let mut batch_encode_start: Option<std::time::Instant> = None;
            let mut iteration_duration: Option<std::time::Duration> = None;

            let mut step = 0usize;
            while step < max_new_tokens {
                let batch_idx = pending_count;

                if batch_idx == 0 {
                    let current_pos = start_pos + prompt_len + step;
                    let remaining = max_new_tokens - step;
                    let lookahead = remaining.min(batch_size);
                    let required_len = current_pos + lookahead;
                    model.ensure_kv_capacity(
                        foundry,
                        bindings,
                        fast_bindings,
                        &mut session.context_config,
                        current_pos,
                        required_len,
                    )?;
                }

                if batch_idx == 0 && !foundry.is_capturing() {
                    foundry.start_capture()?;
                }

                let step_start = std::time::Instant::now();
                if emit_host_metrics && !profiling_per_kernel && batch_idx == 0 {
                    batch_encode_start = Some(step_start);
                }

                let logits = bindings.get(&self.sample_logits_binding)?;
                let mut logits_arg = logits.clone();
                if step == 0 && last_prefill_m > 1 {
                    logits_arg.offset += (last_prefill_m - 1) * (vocab_size as usize) * 2;
                }

                let effective_top_k = if greedy { 1 } else { top_k };
                let sample_out_name = format!("sample_out_{batch_idx}");
                let sample_out_arg = bindings.get(&sample_out_name)?;
                let sample_out = &sample_out_arg;
                let sample_kernel = SampleTopK::new(
                    &logits_arg,
                    sample_out,
                    vocab_size,
                    effective_top_k,
                    top_p,
                    temperature,
                    seed.wrapping_add(step as u32),
                );
                foundry.run(&sample_kernel)?;

                model.set_binding(
                    bindings,
                    fast_bindings,
                    &self.input_ids_binding,
                    (*sample_out).clone(),
                );

                model.set_int_global(bindings, &self.position_offset_key, start_pos + prompt_len + step);
                if self.apply_derived_globals {
                    model.apply_derived_globals(bindings);
                }

                model.forward(foundry, bindings, &*fast_bindings)?;

                pending_count += 1;
                step += 1;

                if pending_count >= batch_size || step == max_new_tokens {
                    if !profiling_per_kernel {
                        let end_capture_start = std::time::Instant::now();
                        let cmd = foundry.end_capture()?;
                        let end_capture_duration = end_capture_start.elapsed();

                        if emit_host_metrics {
                            metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                                parent_op_name: "workflow_loop".to_string(),
                                internal_kernel_name: "end_capture".to_string(),
                                duration_us: end_capture_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                            });
                        }

                        let wait_start = std::time::Instant::now();
                        cmd.wait_until_completed();
                        let wait_duration = wait_start.elapsed();

                        if emit_host_metrics {
                            let mut cb_data = rustc_hash::FxHashMap::default();
                            cb_data.insert("batch_size".to_string(), pending_count.to_string());
                            metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                                op_name: "Workflow/Loop/Forward Step/CB Wait".to_string(),
                                backend: "Foundry".to_string(),
                                duration_us: wait_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                                data: Some(cb_data),
                            });

                            if let Some(start) = batch_encode_start.take() {
                                let total_duration = start.elapsed();
                                iteration_duration = Some(total_duration);
                                let per_token_us = (total_duration.as_micros() / pending_count as u128).min(u128::from(u64::MAX)) as u64;
                                metallic_instrumentation::record_metric_async!(
                                    metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                                        parent_op_name: "workflow_loop".to_string(),
                                        internal_kernel_name: "forward_step_total".to_string(),
                                        duration_us: per_token_us,
                                    }
                                );
                            }
                        }
                    } else {
                        foundry.synchronize()?;
                        if emit_host_metrics {
                            iteration_duration = Some(step_start.elapsed());
                        }
                    }

                    let process_start = std::time::Instant::now();
                    let mut batch_done = false;
                    for i in 0..pending_count {
                        let sample_out_name = format!("sample_out_{i}");
                        let sample_out_arg = bindings.get(&sample_out_name)?;
                        let token_buf = sample_out_arg.buffer.as_ref().unwrap();
                        let token: u32 = token_buf.read_scalar::<u32>();

                        // Execute the host-side stages in spec order for this token.
                        for stage in &self.stages {
                            match stage {
                                WorkflowStageSpec::Sample { .. } => {}
                                WorkflowStageSpec::GraphForward { .. } => {}
                                WorkflowStageSpec::CheckEos { input, output, .. } => {
                                    let tok = if input == &self.sample_token_var {
                                        token
                                    } else {
                                        ctx.values
                                            .get(input)
                                            .and_then(|v| v.as_u32())
                                            .ok_or_else(|| {
                                                MetalError::InvalidOperation(format!(
                                                    "check_eos missing token input '{input}' (u32)"
                                                ))
                                            })?
                                    };
                                    let should_stop = tok == eos_token;
                                    ctx.values.insert(output.clone(), Value::Bool(should_stop));
                                    if should_stop && !ignore_eos_stop {
                                        batch_done = true;
                                    }
                                }
                                WorkflowStageSpec::AppendToken { input, .. } => {
                                    let tok = if input == &self.sample_token_var {
                                        token
                                    } else {
                                        ctx.values
                                            .get(input)
                                            .and_then(|v| v.as_u32())
                                            .ok_or_else(|| {
                                                MetalError::InvalidOperation(format!(
                                                    "append_token missing token input '{input}' (u32)"
                                                ))
                                            })?
                                    };
                                    generated.push(tok);
                                }
                            }

                            if batch_done {
                                break;
                            }
                        }

                        if batch_done {
                            break;
                        }

                        if !on_token(
                            token,
                            prefill_duration,
                            setup_duration,
                            iteration_duration.map(|d| d / pending_count as u32),
                        )? {
                            batch_done = true;
                            break;
                        }
                    }
                    // If the workflow did not append tokens explicitly, treat the sampled tokens as generated.
                    if !self.has_append_stage && pending_count > 0 {
                        for i in 0..pending_count {
                            let sample_out_name = format!("sample_out_{i}");
                            let sample_out_arg = bindings.get(&sample_out_name)?;
                            let token_buf = sample_out_arg.buffer.as_ref().unwrap();
                            generated.push(token_buf.read_scalar::<u32>());
                        }
                    }

                    let process_duration = process_start.elapsed();
                    if emit_host_metrics && !process_duration.is_zero() && !profiling_per_kernel && pending_count > 0 {
                        let per_token_us = (process_duration.as_micros() / pending_count as u128).min(u128::from(u64::MAX)) as u64;
                        metallic_instrumentation::record_metric_async!(
                            metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                                parent_op_name: "workflow_loop".to_string(),
                                internal_kernel_name: "host_process".to_string(),
                                duration_us: per_token_us,
                            }
                        );
                    }

                    pending_count = 0;
                    iteration_duration = None;

                    if batch_done {
                        break;
                    }
                }
            }

            Ok(generated)
        })?;

        ctx.values.insert(self.append_output.clone(), Value::TokensU32(generated.into()));

        Ok(WorkflowOpOutcome::Continue)
    }
}
