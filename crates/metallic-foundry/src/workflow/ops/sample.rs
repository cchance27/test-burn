use half::f16;
use metallic_env::SAMPLE_CPU_FALLBACK;

use crate::{
    error::MetalError, metals::sampling::{ApplyRepetitionPenalty, RepetitionStateIngest, RepetitionStateInit, RepetitionStateUpdateFromToken, SampleTopK}, types::{MetalBuffer, MetalResourceOptions, TensorArg}, workflow::{
        Value, ops::{
            WorkflowOp, WorkflowOpOutcome, common::{
                INTERNAL_DECODE_BATCH_IDX, INTERNAL_DECODE_BATCH_SIZE, INTERNAL_LAST_DECODE_US, err_missing_input, write_internal_usize
            }
        }, runner::WorkflowExecutionContext, spec::Param
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
    // Optional decode-batch ring of per-step outputs (used by WhileBatchedOp).
    out_ring: Vec<(MetalBuffer, TensorArg)>,
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
            out_ring: Vec::new(),
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

fn filter_prompt_tokens_for_penalty<'a>(prompt_tokens: &'a [u32], eos_token: Option<u32>) -> std::borrow::Cow<'a, [u32]> {
    let Some(eos) = eos_token else {
        return std::borrow::Cow::Borrowed(prompt_tokens);
    };
    if !prompt_tokens.contains(&eos) {
        return std::borrow::Cow::Borrowed(prompt_tokens);
    }

    let filtered = prompt_tokens.iter().copied().filter(|tok| *tok != eos).collect::<Vec<_>>();
    std::borrow::Cow::Owned(filtered)
}

fn effective_penalty_window(repeat_last_n: usize, max_tokens: usize) -> usize {
    if repeat_last_n == 0 {
        return 0;
    }
    // When generation is uncapped, a tiny window (e.g. 64) often fails to catch
    // longer semantic loops (paragraph-level cycles). Boost window defensively.
    let boosted = if max_tokens == 0 { repeat_last_n.max(256) } else { repeat_last_n };
    boosted.min(1024)
}

fn sample_cpu_fallback_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| SAMPLE_CPU_FALLBACK.get().ok().flatten().unwrap_or(false))
}

fn sample_cpu_fallback_from_ctx(ctx: &WorkflowExecutionContext<'_>) -> Option<bool> {
    ctx.values.get("sample_cpu_fallback").and_then(|v| v.as_bool())
}

fn read_logits_f32(ctx: &mut WorkflowExecutionContext<'_>, logits_arg: &TensorArg, vocab_size: usize) -> Result<Vec<f32>, MetalError> {
    let Some(src_buf) = logits_arg.buffer.as_ref() else {
        return Err(MetalError::InvalidOperation("SampleOp logits tensor missing buffer".into()));
    };
    let elem_size = logits_arg.dtype.size_bytes();
    let size_bytes = vocab_size
        .checked_mul(elem_size)
        .ok_or_else(|| MetalError::OperationFailed("SampleOp logits size overflow".into()))?;
    let staging = ctx
        .foundry
        .device
        .new_buffer(size_bytes.max(4), MetalResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferCreationFailed(size_bytes.max(4)))?;
    ctx.foundry.blit_copy_sync(src_buf, logits_arg.offset, &staging, 0, size_bytes)?;

    match logits_arg.dtype {
        crate::tensor::Dtype::F16 => {
            let raw = staging.read_to_vec::<u16>(vocab_size);
            Ok(raw.into_iter().map(|bits| f16::from_bits(bits).to_f32()).collect())
        }
        crate::tensor::Dtype::F32 => Ok(staging.read_to_vec::<f32>(vocab_size)),
        other => Err(MetalError::UnsupportedDtype {
            operation: "SampleOp(read_logits_f32)",
            dtype: other,
        }),
    }
}

#[derive(Clone, Copy)]
struct CpuRng {
    state: u32,
}

impl CpuRng {
    fn mix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^ (x >> 16)
    }

    fn seeded(seed: u32) -> Self {
        let v = if seed == 0 { 1 } else { seed };
        let mut state = Self::mix(v);
        if state == 0 {
            state = 1;
        }
        Self { state }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        let bits = (x >> 9) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
}

fn cpu_sample_topk_topp(logits: &[f32], top_k: u32, top_p: f32, min_p: f32, temperature: f32, seed: u32) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut fallback_idx = 0usize;
    let mut fallback_found = false;
    let mut fallback_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && (!fallback_found || v > fallback_val || (v == fallback_val && i < fallback_idx)) {
            fallback_idx = i;
            fallback_val = v;
            fallback_found = true;
        }
    }
    let fallback = if fallback_found { fallback_idx as u32 } else { 0 };

    let denom = temperature.max(1.0e-6);
    let inv_t = 1.0f32 / denom;
    let k = if top_k == 0 { 1usize } else { (top_k as usize).min(logits.len()) };
    if k == 0 {
        return fallback;
    }

    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if !v.is_finite() {
                return None;
            }
            Some((i as u32, v * inv_t))
        })
        .collect();

    if candidates.is_empty() {
        return fallback;
    }

    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    candidates.truncate(k);

    let maxv = candidates[0].1;
    let mut probs: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
    let mut sum = 0.0f32;
    for (idx, v) in candidates {
        let e = (v - maxv).exp();
        if e.is_finite() && e > 0.0 {
            probs.push((idx, e));
            sum += e;
        }
    }

    if probs.is_empty() || !sum.is_finite() || sum <= 0.0 {
        return fallback;
    }

    let top_p = if top_p.is_finite() { top_p.clamp(0.0, 1.0) } else { 1.0 };
    let min_p = if min_p.is_finite() { min_p.clamp(0.0, 1.0) } else { 0.0 };
    let max_prob = probs[0].1 / sum;
    let min_p_cut = if min_p > 0.0 { min_p * max_prob } else { 0.0 };

    let mut filtered: Vec<(u32, f32)> = Vec::with_capacity(probs.len());
    let mut cumulative = 0.0f32;
    for (idx, e) in probs {
        let p = e / sum;
        if min_p_cut > 0.0 && p < min_p_cut {
            continue;
        }
        filtered.push((idx, p));
        cumulative += p;
        if cumulative >= top_p {
            break;
        }
    }

    if filtered.is_empty() {
        return fallback;
    }

    let renorm = if cumulative.is_finite() && cumulative > 0.0 {
        1.0f32 / cumulative
    } else {
        1.0
    };
    for (_, p) in &mut filtered {
        *p *= renorm;
    }

    let mut rng = CpuRng::seeded(seed);
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    let mut chosen = filtered[0].0;
    for (idx, p) in filtered {
        acc += p;
        if r <= acc {
            chosen = idx;
            break;
        }
    }
    chosen
}

fn debug_log_sample(step: u32, sampled: u32, logits: &[f32], top_n: usize, eos_token: Option<u32>) {
    let mut finite: Vec<(usize, f32)> = logits.iter().copied().enumerate().filter(|(_, v)| v.is_finite()).collect();
    finite.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let best = finite
        .iter()
        .take(top_n)
        .map(|(idx, val)| format!("{idx}:{val:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    let nan_count = logits.iter().filter(|v| v.is_nan()).count();
    let inf_count = logits.iter().filter(|v| v.is_infinite()).count();
    let eos_diag = eos_token
        .and_then(|eos| usize::try_from(eos).ok())
        .and_then(|eos_idx| logits.get(eos_idx).copied().map(|eos_logit| (eos_idx, eos_logit)))
        .map(|(eos_idx, eos_logit)| {
            let eos_rank = finite.iter().position(|(idx, _)| *idx == eos_idx).map(|r| r + 1).unwrap_or(0);
            format!("eos_idx={eos_idx} eos_logit={eos_logit:.4} eos_rank={eos_rank}")
        })
        .unwrap_or_else(|| "eos_idx=<none>".to_string());
    tracing::debug!(
        "SampleOp debug step={} sampled={} nan={} inf={} {} top{}=[{}]",
        step,
        sampled,
        nan_count,
        inf_count,
        eos_diag,
        top_n,
        best
    );
}

impl WorkflowOp for SampleOp {
    fn begin_run(&mut self, _ctx: &mut WorkflowExecutionContext<'_>) -> Result<(), MetalError> {
        // Seed offset should restart per workflow invocation (per user turn).
        self.step = 0;
        // Repetition state must restart per run; keeping it across runs can over-penalize
        // tokens and destabilize generation into loops/rambling.
        self.rep_initialized = false;
        self.rep_needs_prompt_ingest = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.step = 0;
        self.rep_initialized = false;
        self.rep_needs_prompt_ingest = true;
    }

    fn execute(
        &mut self,
        ctx: &mut WorkflowExecutionContext<'_>,
        _on_token: &mut dyn FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    ) -> Result<WorkflowOpOutcome, MetalError> {
        let decode_start = std::time::Instant::now();
        let logits_arg = ctx
            .values
            .get(&self.logits_var)
            .and_then(|v| v.as_tensor())
            .cloned()
            .ok_or_else(|| err_missing_input("SampleOp", &self.logits_var, "Tensor"))?;

        let temp = ctx.resolve_param_f32(&self.temperature)?;
        let top_k = ctx.resolve_param_u32(&self.top_k)?;
        let top_p = ctx.resolve_param_f32(&self.top_p)?;
        let min_p = ctx.resolve_param_f32(&self.min_p)?;
        let repeat_penalty = ctx.resolve_param_f32(&self.repeat_penalty)?;
        let repeat_last_n = ctx.resolve_param_usize(&self.repeat_last_n)?;
        let presence_penalty = ctx.resolve_param_f32(&self.presence_penalty)?;
        let frequency_penalty = ctx.resolve_param_f32(&self.frequency_penalty)?;
        let max_tokens = ctx.values.get("max_tokens").and_then(|v| v.as_usize()).unwrap_or(0);
        let eos_token = ctx.values.get("eos_token").and_then(|v| v.as_u32());
        let seed = ctx.resolve_param_u32(&self.seed)?.wrapping_add(self.step);

        self.step = self.step.wrapping_add(1);

        let in_batched_decode_loop = ctx.values.contains_key(INTERNAL_DECODE_BATCH_SIZE);
        let decode_batch_size = ctx
            .values
            .get(INTERNAL_DECODE_BATCH_SIZE)
            .and_then(|v| v.as_usize())
            .unwrap_or(1)
            .max(1);
        let decode_batch_idx = ctx.values.get(INTERNAL_DECODE_BATCH_IDX).and_then(|v| v.as_usize()).unwrap_or(0);

        // Allocate and reuse output buffers. In batched decode, we need one buffer per iteration
        // so the loop can read back all tokens after the capture completes.
        let (out_buffer, out_arg) = if in_batched_decode_loop {
            if self.out_ring.len() != decode_batch_size {
                self.out_ring.clear();
                self.out_ring.reserve(decode_batch_size);
                for _ in 0..decode_batch_size {
                    let buf = ctx
                        .foundry
                        .device
                        .new_buffer(4, MetalResourceOptions::StorageModeShared)
                        .expect("Failed to allocate sample output buffer");
                    let arg = TensorArg::from_buffer(buf.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
                    self.out_ring.push((buf, arg));
                }
            }
            let idx = decode_batch_idx.min(decode_batch_size.saturating_sub(1));
            let (buf, arg) = &self.out_ring[idx];
            (buf, arg)
        } else {
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
            (
                self.out_buffer.as_ref().expect("out_buffer set"),
                self.out_arg.as_ref().expect("out_arg set"),
            )
        };

        let vocab_size = logits_arg.dims()[logits_arg.dims().len() - 1] as u32;

        // Apply token penalties in-place to logits before sampling (GPU-side state).
        let use_token_penalties = (repeat_penalty > 1.0 && repeat_penalty.is_finite())
            || (presence_penalty > 0.0 && presence_penalty.is_finite())
            || (frequency_penalty > 0.0 && frequency_penalty.is_finite());

        let window_len = effective_penalty_window(repeat_last_n, max_tokens);
        if use_token_penalties && window_len > 0 {
            // (Re)allocate state buffers if needed.
            if self.rep_ring_buf.is_none() || self.rep_window_len != window_len {
                tracing::debug!(
                    "SampleOp repetition window configured: requested={}, max_tokens={}, effective={}",
                    repeat_last_n,
                    max_tokens,
                    window_len
                );
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
                    let prompt_tokens = filter_prompt_tokens_for_penalty(prompt_tokens, eos_token);
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
                        buf.copy_from_slice(prompt_tokens.as_ref());
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
                &logits_arg,
                pairs_arg,
                vocab_size,
                window_len as u32,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
            );
            ctx.foundry.run(&penalty_kernel)?;
        }

        let debug_logits = metallic_instrumentation::logging::debug_sample_logits_enabled()
            && self.step <= metallic_instrumentation::logging::debug_sample_logits_max_steps()
            && !in_batched_decode_loop;
        let cpu_fallback = sample_cpu_fallback_from_ctx(ctx).unwrap_or_else(sample_cpu_fallback_enabled) && !in_batched_decode_loop;
        let logits_snapshot = if debug_logits || cpu_fallback {
            Some(read_logits_f32(ctx, &logits_arg, vocab_size as usize)?)
        } else {
            None
        };

        if cpu_fallback {
            let logits = logits_snapshot.as_ref().expect("logits snapshot set");
            let token = cpu_sample_topk_topp(logits, top_k, top_p, min_p, temp, seed);
            out_buffer.copy_from_slice(&[token]);
        } else {
            let kernel = SampleTopK::new(&logits_arg, out_arg, vocab_size, top_k, top_p, min_p, temp, seed);
            ctx.foundry.run(&kernel)?;
        }

        // Update repetition state with the newly sampled token so the next step sees it.
        if use_token_penalties && self.rep_window_len > 0 {
            let ring_arg = self.rep_ring_arg.as_ref().expect("rep_ring_arg set");
            let pairs_arg = self.rep_pairs_arg.as_ref().expect("rep_pairs_arg set");
            let meta_arg = self.rep_meta_arg.as_ref().expect("rep_meta_arg set");
            let upd = RepetitionStateUpdateFromToken::new(ring_arg, pairs_arg, meta_arg, out_arg, self.rep_window_len as u32);
            ctx.foundry.run(&upd)?;
        }

        // In batched decode, avoid scalar readback here; WhileBatchedOp reads the tensor buffers after
        // the capture completes. In non-batched mode, keep compatibility with existing workflows.
        if in_batched_decode_loop {
            ctx.values.insert(self.output_var.clone(), Value::Tensor(out_arg.clone()));
        } else {
            let token = out_buffer.read_scalar::<u32>();
            if debug_logits && let Some(logits) = logits_snapshot.as_ref() {
                debug_log_sample(
                    self.step,
                    token,
                    logits,
                    metallic_instrumentation::logging::debug_sample_logits_top_n(),
                    eos_token,
                );
            }
            ctx.values.insert(self.output_var.clone(), Value::U32(token));
        }
        write_internal_usize(ctx, INTERNAL_LAST_DECODE_US, decode_start.elapsed().as_micros() as usize);

        Ok(WorkflowOpOutcome::Continue)
    }
}

#[path = "sample.test.rs"]
mod tests;
