use std::{
    collections::BTreeMap, env, fs::OpenOptions, io::Write, path::PathBuf, sync::{Arc, mpsc}, time::{Duration, Instant}
};

use metallic_cli_helpers::app_event::AppEvent;
use metallic_instrumentation::{MetricEvent, global_cached_memory_profiler, record_metric_async};
use rand::prelude::*;
use rustc_hash::FxHashMap;

use super::{Context, MetalError, SamplerBuffers, Tensor};
use crate::{
    TensorElement, Tokenizer, caching::CacheMetrics, kernels::{embedding_lookup::EmbeddingLookupOp, sample_topk_topp::SampleTopKTopPOp}, models::qwen25::Qwen25, operation::CommandBuffer, tensor::dtypes::U32
};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

const METALLIC_LOG_CACHE_STATS_ENV: &str = "METALLIC_LOG_CACHE_STATS";
const METALLIC_LOG_CACHE_STATS_DEFAULT_FILE: &str = "metal-cache-stats.log";

fn embed_single_token_cached<T: TensorElement>(qwen: &Qwen25<T>, token_id: u32, ctx: &mut Context<T>) -> Result<Tensor<T>, MetalError> {
    let mut indices = ctx.shared_u32_host_tensor(1)?;
    indices.as_mut_slice()[0] = token_id;
    let out = ctx.call::<EmbeddingLookupOp>((&qwen.embed_weight, &indices), None)?;
    debug_assert_eq!(out.dims(), &[1, 1, qwen.config.d_model]);
    Ok(out)
}

struct CacheStatsLogger {
    file: Option<std::sync::Mutex<std::fs::File>>,
}

impl CacheStatsLogger {
    fn from_env() -> Self {
        let path = env::var(METALLIC_LOG_CACHE_STATS_ENV).ok().and_then(resolve_cache_log_path);

        let file = path.and_then(|path| match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(std::sync::Mutex::new(file)),
            Err(err) => {
                eprintln!("Failed to open cache stats log at {:?}: {}", path, err);
                None
            }
        });

        Self { file }
    }

    fn enabled(&self) -> bool {
        self.file.is_some()
    }

    fn log_line(&self, line: &str) {
        let Some(file) = &self.file else {
            return;
        };

        if let Ok(mut file) = file.lock()
            && let Err(err) = writeln!(file, "{line}")
        {
            eprintln!("Failed to write cache stats log: {err}");
        }
    }
}

fn resolve_cache_log_path(value: String) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lowered = trimmed.to_ascii_lowercase();
    if matches!(lowered.as_str(), "0" | "false" | "no" | "off") {
        return None;
    }

    if matches!(lowered.as_str(), "1" | "true" | "yes" | "on") {
        return Some(PathBuf::from(METALLIC_LOG_CACHE_STATS_DEFAULT_FILE));
    }

    Some(PathBuf::from(trimmed))
}

fn cache_stats_logger() -> &'static CacheStatsLogger {
    static LOGGER: std::sync::OnceLock<CacheStatsLogger> = std::sync::OnceLock::new();
    LOGGER.get_or_init(CacheStatsLogger::from_env)
}

fn cache_stats_logging_enabled() -> bool {
    cache_stats_logger().enabled()
}

fn log_cache_stats<T: TensorElement>(ctx: &Context<T>, phase: &str, step: usize) {
    if !cache_stats_logging_enabled() {
        return;
    }

    let line = match ctx.get_cache_stats() {
        Some(stats) => {
            let segments = [
                describe_cache_metrics("gemm", &stats.gemm),
                describe_cache_metrics("descriptor", &stats.descriptor),
                describe_cache_metrics("softmax", &stats.softmax),
                describe_cache_metrics("sdpa", &stats.sdpa),
            ];
            format!("[metal-cache] {phase}#{step}: {}", segments.join(" "))
        }
        None => format!("[metal-cache] {phase}#{step}: cache-uninitialized"),
    };

    cache_stats_logger().log_line(&line);
}

fn describe_cache_metrics(name: &str, metrics: &CacheMetrics) -> String {
    let last = metrics
        .last_event
        .as_ref()
        .map(|event| event.to_string())
        .unwrap_or_else(|| "none".to_string());
    let requests = metrics.hits + metrics.misses;
    let hit_rate = if requests > 0 {
        (metrics.hits as f64 / requests as f64) * 100.0
    } else {
        0.0
    };

    let oldest = format_duration(metrics.oldest_entry_age);
    let newest = format_duration(metrics.newest_entry_age);
    let lru_idle = format_duration(metrics.longest_idle_age);
    let mru_idle = format_duration(metrics.shortest_idle_age);
    let reuse = metrics
        .max_entry_reuse_count
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());

    format!(
        "{name}(size={size} hits={hits} misses={misses} evict={evict} requests={requests} hit_rate={hit_rate:.1}% oldest={oldest} newest={newest} lru_idle={lru_idle} mru_idle={mru_idle} reuse_max={reuse} last={last})",
        size = metrics.size,
        hits = metrics.hits,
        misses = metrics.misses,
        evict = metrics.evictions,
        requests = requests,
        oldest = oldest,
        newest = newest,
        lru_idle = lru_idle,
        mru_idle = mru_idle,
        reuse = reuse,
        last = last,
        name = name,
        hit_rate = hit_rate
    )
}

fn format_duration(duration: Option<Duration>) -> String {
    match duration {
        Some(value) => {
            if value.as_secs_f64() >= 1.0 {
                format!("{:.2}s", value.as_secs_f64())
            } else if value.as_millis() >= 1 {
                format!("{:.2}ms", value.as_secs_f64() * 1e3)
            } else {
                format!("{:.0}us", value.as_secs_f64() * 1e6)
            }
        }
        None => "-".to_string(),
    }
}

/// Generation configuration (defaults chosen by user)
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Initial KV cache headroom in tokens beyond the current prompt length.
    /// This lets us avoid over-allocating the KV pool when typical generations are short.
    /// If generation exceeds this, we currently do not grow the KV cache mid-run.
    pub kv_initial_headroom_tokens: usize,
    /// Random seed for sampling. If None, a random seed will be generated.
    pub seed: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
            kv_initial_headroom_tokens: 256,
            seed: None,
        }
    }
}

/// Deterministic fallback that inspects only the final logits row.
/// NOTE: Forces a GPU sync so call sparingly on hot paths.
fn greedy_argmax_last_token<T: TensorElement>(
    logits_tensor: &Tensor<T>,
    vocab_size: usize,
    ctx: &mut Context<T>,
) -> Result<u32, MetalError> {
    if vocab_size == 0 {
        return Ok(0);
    }

    ctx.synchronize();

    let dims = logits_tensor.dims();
    if dims.is_empty() {
        return Err(MetalError::InvalidShape("Logits tensor must have at least one dimension".into()));
    }

    let last_dim = *dims.last().unwrap();
    if last_dim != vocab_size {
        return Err(MetalError::InvalidShape(format!(
            "Logits tensor last dimension {} does not match vocab size {}",
            last_dim, vocab_size
        )));
    }

    let slice = logits_tensor.as_slice();
    if slice.len() < vocab_size {
        return Err(MetalError::InvalidShape("Logits tensor shorter than vocab size".to_string()));
    }

    let start_idx = slice.len() - vocab_size;
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let mut found = false;

    for (offset, &raw) in slice[start_idx..].iter().enumerate() {
        let value = T::to_f32(raw);
        if !value.is_finite() {
            continue;
        }

        let token_idx = offset as u32;
        if !found || value > best_val || (value == best_val && token_idx > best_idx) {
            found = true;
            best_val = value;
            best_idx = token_idx;
        }
    }

    Ok(if found { best_idx } else { 0 })
}

struct SampleResult {
    token: u32,
    sample_duration: Duration,
    iteration_duration: Duration,
    wait_duration: Duration,
}

pub(crate) struct GpuSampleFuture {
    token_tensor: Option<Tensor<U32>>,
    command_buffer: Option<CommandBuffer>,
    iteration_start: Instant,
    sample_start: Instant,
    submitted: bool,
    ready_token: Option<u32>,
    ready_sample_duration: Duration,
}

impl GpuSampleFuture {
    fn pending(token_tensor: Tensor<U32>, command_buffer: CommandBuffer, iteration_start: Instant, sample_start: Instant) -> Self {
        Self {
            token_tensor: Some(token_tensor),
            command_buffer: Some(command_buffer),
            iteration_start,
            sample_start,
            submitted: false,
            ready_token: None,
            ready_sample_duration: Duration::ZERO,
        }
    }

    fn ready(token: u32, iteration_start: Instant, sample_duration: Duration) -> Self {
        Self {
            token_tensor: None,
            command_buffer: None,
            iteration_start,
            sample_start: iteration_start,
            submitted: true,
            ready_token: Some(token),
            ready_sample_duration: sample_duration,
        }
    }

    fn mark_submitted(&mut self) {
        if self.submitted {
            return;
        }
        self.sample_start = Instant::now();
        self.submitted = true;
    }

    fn finalize(&mut self) -> Result<SampleResult, MetalError> {
        let token_tensor = self.token_tensor.take().expect("pending GPU sample must retain output tensor");
        let token_slice = token_tensor.as_slice();
        let now = Instant::now();
        let sample_duration = now.saturating_duration_since(self.sample_start);
        let iteration_duration = now.saturating_duration_since(self.iteration_start);
        let token = token_slice[0];
        self.ready_token = Some(token);
        self.ready_sample_duration = sample_duration;
        Ok(SampleResult {
            token,
            sample_duration,
            iteration_duration,
            wait_duration: Duration::ZERO,
        })
    }

    fn wait<T: TensorElement>(mut self, ctx: &mut Context<T>) -> Result<SampleResult, MetalError> {
        if let Some(token) = self.ready_token {
            return Ok(SampleResult {
                token,
                sample_duration: self.ready_sample_duration,
                iteration_duration: self.iteration_start.elapsed(),
                wait_duration: Duration::ZERO,
            });
        }

        let wait_start = Instant::now();
        if let Some(command_buffer) = self.command_buffer.take() {
            let completions = ctx.wait_for_command_buffer(command_buffer, None);
            ctx.process_pipeline_completions(completions);
        }

        if let Some(token_tensor) = self.token_tensor.as_ref() {
            token_tensor.defining_cmd_buffer.borrow_mut().take();
        }

        let wait_duration = wait_start.elapsed();
        let mut result = self.finalize()?;
        result.wait_duration = wait_duration;
        Ok(result)
    }
}

/// GPU-based sample from logits using top-k and top-p (nucleus) sampling with asynchronous completion.
pub(crate) fn gpu_sample_top_k_top_p_async<T: TensorElement>(
    logits_tensor: &Tensor<T>,
    vocab_size: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: Option<u32>,
    ctx: &mut Context<T>,
    iteration_start: Instant,
) -> Result<GpuSampleFuture, MetalError> {
    let sample_start = Instant::now();

    if temperature <= 0.0 || !temperature.is_finite() || top_k == 0 {
        let token = greedy_argmax_last_token(logits_tensor, vocab_size, ctx)?;
        return Ok(GpuSampleFuture::ready(token, iteration_start, sample_start.elapsed()));
    }

    // Use the provided per-step seed (preferred for deterministic generation),
    // or generate a fresh one when the caller doesn't care about reproducibility.
    let seed = seed.unwrap_or_else(|| rand::rng().next_u32());

    let (output_token,) = ctx.call_custom::<SampleTopKTopPOp>(
        (
            logits_tensor.clone(),
            vocab_size as u32,
            top_k as u32,
            top_p,
            temperature,
            seed,
            40u32,
        ),
        None,
    )?;

    let command_buffer = output_token
        .defining_cmd_buffer
        .borrow()
        .clone()
        .expect("sampling output tensor must have defining command buffer");

    Ok(GpuSampleFuture::pending(
        output_token,
        command_buffer,
        iteration_start,
        sample_start,
    ))
}

pub fn gpu_sample_top_k_top_p<T: TensorElement>(
    logits_tensor: &Tensor<T>,
    vocab_size: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: Option<u32>,
    ctx: &mut Context<T>,
) -> Result<u32, MetalError> {
    let mut future = gpu_sample_top_k_top_p_async(logits_tensor, vocab_size, top_k, top_p, temperature, seed, ctx, Instant::now())?;
    ctx.submit_active_command_buffer();
    future.mark_submitted();
    ctx.poll_command_buffer_completions();
    let result = future.wait(ctx)?;
    Ok(result.token)
}

struct CompletedToken {
    token_id: u32,
    sample_duration: Duration,
    iteration_duration: Duration,
    sample_wait_duration: Duration,
}

impl CompletedToken {
    fn from_sample(sample: SampleResult) -> Self {
        Self {
            token_id: sample.token,
            sample_duration: sample.sample_duration,
            iteration_duration: sample.iteration_duration,
            sample_wait_duration: sample.wait_duration,
        }
    }
}

struct ProcessOutcome {
    continue_generation: bool,
    is_eos: bool,
}

fn process_completed_token<T: TensorElement, F>(
    token: CompletedToken,
    generated_ids: &mut Vec<u32>,
    tokenizer: &Tokenizer,
    decoded_chunk: &mut String,
    decode_scratch: &mut Vec<u8>,
    ctx: &mut Context<T>,
    metric_recording_overhead: &mut Duration,
    token_callback: &mut F,
) -> Result<ProcessOutcome, MetalError>
where
    F: FnMut(u32, Arc<str>, Duration) -> Result<bool, MetalError>,
{
    let sample_metric_start = Instant::now();
    record_metric_async!(MetricEvent::InternalKernelCompleted {
        parent_op_name: "sampling".to_string(),
        internal_kernel_name: "next_token_ready_latency".to_string(),
        duration_us: token.sample_duration.as_micros().max(1) as u64,
    });
    *metric_recording_overhead += sample_metric_start.elapsed();

    if !token.sample_wait_duration.is_zero() {
        let metric_start = Instant::now();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "sampling".to_string(),
            internal_kernel_name: "next_token_wait".to_string(),
            duration_us: token.sample_wait_duration.as_micros().max(1) as u64,
        });
        *metric_recording_overhead += metric_start.elapsed();
    }

    if !token.iteration_duration.is_zero() {
        let metric_start = Instant::now();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "generation_loop".to_string(),
            internal_kernel_name: "iteration_total".to_string(),
            duration_us: token.iteration_duration.as_micros() as u64,
        });
        *metric_recording_overhead += metric_start.elapsed();
    }

    let push_start = Instant::now();
    generated_ids.push(token.token_id);
    let push_duration = push_start.elapsed();
    if !push_duration.is_zero() {
        let metric_start = Instant::now();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "generation_loop".to_string(),
            internal_kernel_name: "token_push".to_string(),
            duration_us: push_duration.as_micros() as u64,
        });
        *metric_recording_overhead += metric_start.elapsed();
    }

    let decode_start = Instant::now();
    let decoded_piece = tokenizer.decode_token_arc(token.token_id, decoded_chunk, decode_scratch)?;
    let decode_duration = decode_start.elapsed();
    let metric_start = Instant::now();
    record_metric_async!(MetricEvent::InternalKernelCompleted {
        parent_op_name: "decoding".to_string(),
        internal_kernel_name: "token_decode".to_string(),
        duration_us: decode_duration.as_micros().max(1) as u64,
    });
    *metric_recording_overhead += metric_start.elapsed();

    let cache_log_start = Instant::now();
    log_cache_stats(ctx, "generate", generated_ids.len());
    let cache_log_duration = cache_log_start.elapsed();
    if !cache_log_duration.is_zero() {
        let metric_start = Instant::now();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "generation_loop".to_string(),
            internal_kernel_name: "cache_logging".to_string(),
            duration_us: cache_log_duration.as_micros() as u64,
        });
        *metric_recording_overhead += metric_start.elapsed();
    }

    let mut continue_generation = true;
    if let Some(piece) = decoded_piece {
        let callback_start = Instant::now();
        continue_generation = token_callback(token.token_id, piece, token.iteration_duration)?;
        let callback_duration = callback_start.elapsed();
        if !callback_duration.is_zero() {
            let metric_start = Instant::now();
            record_metric_async!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "generation_loop".to_string(),
                internal_kernel_name: "token_callback".to_string(),
                duration_us: callback_duration.as_micros() as u64,
            });
            *metric_recording_overhead += metric_start.elapsed();
        }
    }

    let eos_check_start = Instant::now();
    let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
    let is_eos = token.token_id == eos_token_id;
    let eos_check_duration = eos_check_start.elapsed();
    if !eos_check_duration.is_zero() {
        let metric_start = Instant::now();
        record_metric_async!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "generation_loop".to_string(),
            internal_kernel_name: "eos_check".to_string(),
            duration_us: eos_check_duration.as_micros() as u64,
        });
        *metric_recording_overhead += metric_start.elapsed();
    }

    Ok(ProcessOutcome {
        continue_generation,
        is_eos,
    })
}

/// Sample from logits using top-k and top-p (nucleus) sampling.
/// - `logits` is a slice of f32 representing vocabulary logits.
///   Returns selected token index.
pub fn sample_top_k_top_p<T: TensorElement>(
    logits: &[T::Scalar],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    buffers: &mut SamplerBuffers,
) -> usize {
    let mut fallback_idx = 0usize;
    let mut fallback_found = false;
    let mut fallback_val = f32::NEG_INFINITY;
    for (i, &raw) in logits.iter().enumerate() {
        let val = T::to_f32(raw);
        if val.is_finite() && (!fallback_found || val > fallback_val || (val == fallback_val && i > fallback_idx)) {
            fallback_idx = i;
            fallback_val = val;
            fallback_found = true;
        }
    }

    // Handle deterministic (greedy) sampling when temperature is zero or non-finite.
    if temperature <= 0.0 || !temperature.is_finite() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    if logits.is_empty() {
        return 0;
    }

    if top_k == 0 {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let effective_top_k = std::cmp::min(top_k.max(1), logits.len());

    let scaled = &mut buffers.scaled;
    scaled.clear();
    if scaled.capacity() < effective_top_k {
        scaled.reserve(effective_top_k);
    }

    let indices = &mut buffers.indices;
    indices.clear();
    if indices.capacity() < effective_top_k {
        indices.reserve(effective_top_k);
    }

    for (i, &raw) in logits.iter().enumerate() {
        let val = T::to_f32(raw);
        let scaled_val = val / temperature;
        if !scaled_val.is_finite() {
            continue;
        }

        let insert_pos = scaled.partition_point(|&existing| existing > scaled_val);
        if indices.len() < effective_top_k {
            scaled.insert(insert_pos, scaled_val);
            indices.insert(insert_pos, i);
        } else if insert_pos < effective_top_k {
            scaled.insert(insert_pos, scaled_val);
            indices.insert(insert_pos, i);
            scaled.pop();
            indices.pop();
        }
    }

    if indices.is_empty() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let mut has_positive = false;
    let max_val = scaled[0];
    let mut total = 0.0f32;
    for val in scaled.iter_mut() {
        if val.is_finite() {
            let mut exp_val = (*val - max_val).exp();
            if exp_val > 1e10 {
                exp_val = 1e10;
            } else if exp_val < 1e-10 {
                exp_val = 0.0;
            }
            *val = exp_val;
        } else {
            *val = 0.0;
        }
        total += *val;
        has_positive |= *val > 0.0;
    }

    if !has_positive || total <= 0.0 || total.is_infinite() || total.is_nan() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let normalized_top_p = if top_p.is_finite() { top_p.clamp(0.0, 1.0) } else { 1.0 };
    let mut cutoff = indices.len() - 1;
    if normalized_top_p <= 0.0 {
        cutoff = 0;
    } else if normalized_top_p < 1.0 {
        let mut cum = 0.0f32;
        let threshold = normalized_top_p * total;
        for (i, &weight) in scaled.iter().enumerate() {
            cum += weight;
            cutoff = i;
            if cum >= threshold || cum.is_infinite() || cum.is_nan() {
                break;
            }
        }
    }

    scaled.truncate(cutoff + 1);
    indices.truncate(cutoff + 1);

    let shortlist_total: f32 = scaled.iter().sum();
    if shortlist_total <= 0.0 || shortlist_total.is_infinite() || shortlist_total.is_nan() {
        return indices.first().copied().unwrap_or(if fallback_found { fallback_idx } else { 0 });
    }

    for weight in scaled.iter_mut() {
        *weight /= shortlist_total;
    }

    // Sample using RNG (use simple rng.next_u32() -> float to avoid trait issues)
    let mut rng = rand::rng();
    let r = (rng.next_u32() as f32) / (u32::MAX as f32);
    let mut acc = 0.0f32;
    for (&idx, &prob) in indices.iter().zip(scaled.iter()) {
        acc += prob;
        if r <= acc || acc.is_infinite() || acc.is_nan() {
            return idx;
        }
    }
    indices.last().copied().unwrap_or(if fallback_found { fallback_idx } else { 0 })
}

/// High-level end-to-end generation pipeline that combines tokenization, embedding,
/// model inference, and sampling into a complete inference loop.
pub fn generate<T: TensorElement>(
    qwen: &mut Qwen25<T>,
    tokenizer: &Tokenizer,
    ctx: &mut Context<T>,
    prompt: &str,
    cfg: &GenerationConfig,
) -> Result<String, MetalError> {
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );
    let input_ids = tokenizer.encode(&full_prompt)?;
    let output_tokens = generate_autoregressive_with_kv_cache(qwen, tokenizer, ctx, &input_ids, cfg)?;
    let output_text = tokenizer.decode(&output_tokens)?;
    Ok(output_text)
}

/// High-level autoregressive generation loop using Qwen25 with KV caches for debugging.
/// This implementation processes the full context each time for comparison.
pub fn generate_autoregressive_with_kv_cache<T: TensorElement>(
    qwen: &mut Qwen25<T>,
    tokenizer: &Tokenizer,
    ctx: &mut Context<T>,
    input_ids: &[u32],
    cfg: &GenerationConfig,
) -> Result<Vec<u32>, MetalError> {
    let mut generated_ids = vec![];

    // Create a dummy sender for the streaming function since we don't actually want to stream
    let (tx, _rx) = mpsc::channel();

    // Use the streaming function to handle the generation loop
    let mut callback = |token_id: u32, _decoded_token: Arc<str>, _iteration_duration: Duration| -> Result<bool, MetalError> {
        generated_ids.push(token_id);
        Ok(true) // Continue generation
    };

    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, input_ids, cfg, &mut callback, &tx)?;

    Ok(generated_ids)
}

/// High-level end-to-end generation pipeline with token streaming support
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<T: TensorElement>(
    qwen: &mut Qwen25<T>,
    tokenizer: &Tokenizer,
    ctx: &mut Context<T>,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    // Build full prompt string following Qwen2.5 chat template
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );

    let generation_start = Instant::now();
    let prompt_start = Instant::now();

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    let mut prompt_processing_duration: Option<Duration> = None;
    let mut token_callback = |_token_id, decoded_token: Arc<str>, iteration_duration: Duration| -> Result<bool, MetalError> {
        let now = Instant::now();

        let prompt_duration = *prompt_processing_duration.get_or_insert_with(|| now.duration_since(prompt_start));

        if tx
            .send(AppEvent::Token {
                text: decoded_token,
                prompt_processing: prompt_duration,
                iteration: (!iteration_duration.is_zero()).then_some(iteration_duration),
            })
            .is_err()
        {
            return Ok(false); // Stop generation if UI thread has disconnected
        }
        Ok(true)
    };

    // Generate tokens using the new KV cache approach with streaming
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, &input_ids, cfg, &mut token_callback, tx)?;

    // Send generation completion event with total generation time
    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });

    Ok(())
}

/// High-level autoregressive generation loop with streaming support using Qwen25 with KV Caching.
#[allow(clippy::too_many_arguments)]
pub fn generate_autoregressive_with_kv_cache_streaming<F, T: TensorElement>(
    qwen: &mut Qwen25<T>,
    tokenizer: &Tokenizer,
    ctx: &mut Context<T>,
    input_ids: &[u32],
    cfg: &GenerationConfig,
    token_callback: &mut F,
    _tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError>
where
    F: FnMut(u32, Arc<str>, Duration) -> Result<bool, MetalError>,
{
    // Pre-allocate KV cache for all layers
    let n_layers = qwen.config.n_layers;
    let seq_len = qwen.config.seq_len;
    let n_kv_heads = qwen.config.n_kv_heads;
    let d_model = qwen.config.d_model;
    let n_heads = qwen.config.n_heads;
    let kv_dim = d_model * n_kv_heads / n_heads;
    let kv_head_dim = kv_dim / n_kv_heads;
    let batch_size = 1; // Assuming batch size of 1 for now
    // Default: allocate KV capacity up to prompt_len + max_tokens.
    // We still report pool capacity precisely, so total pool size will reflect bytes actually needed, not a fixed chunk like 640MB.
    let kv_capacity = (input_ids.len().max(1) + cfg.max_tokens).min(seq_len);

    // Determine whether existing cache-backed descriptors must be invalidated because
    // their shapes changed (e.g. when generating with a longer context than before).
    let repeated_batch_heads = batch_size * n_heads;
    let mut cache_shapes_changed = false;
    if !ctx.kv_caches().is_empty() {
        for layer_idx in 0..n_layers {
            match ctx.kv_caches().get(&layer_idx) {
                Some(entry) => {
                    let k_dims = entry.k.dims();
                    if entry.capacity != kv_capacity || k_dims[0] != repeated_batch_heads || k_dims[2] != kv_head_dim {
                        cache_shapes_changed = true;
                        break;
                    }
                }
                None => {
                    cache_shapes_changed = true;
                    break;
                }
            }
        }
    }

    // Ensure KV caches start from a clean slate between generations.
    ctx.clear_kv_caches();
    ctx.kv_cache_pool_mut().reset();

    // Pre-reserve KV cache pool exactly to avoid chunk overshoot
    let bytes_per_element = std::mem::size_of::<T>();
    let per_layer_bytes = (batch_size * n_heads) * kv_capacity * kv_head_dim * bytes_per_element;
    let total_bytes = per_layer_bytes
        .checked_mul(2) // K + V
        .and_then(|b| b.checked_mul(n_layers))
        .ok_or(MetalError::OutOfMemory)?;
    ctx.kv_cache_pool_mut().reserve_exact(total_bytes)?;

    // Report KV cache allocation - only if metrics are enabled
    let kv_cache_size = kv_capacity * batch_size * n_heads * kv_head_dim * std::mem::size_of::<T>();

    let mut forward_pass_breakdown = BTreeMap::new();

    // Only perform memory profiling if we've enabled profiling (meaning metrics are enabled)
    if let Ok(config) = metallic_instrumentation::config::AppConfig::get_or_init_from_env()
        && config.enable_profiling
    {
        // Get current process memory usage using the cached profiler
        let process_memory_bytes = global_cached_memory_profiler().get_process_memory_usage();

        record_metric_async!(MetricEvent::HostMemory {
            total_bytes: process_memory_bytes,
            tensor_pool_reserved_bytes: ctx.pool.total_capacity() as u64,
            tensor_pool_used_bytes: ctx.pool.used_bytes() as u64,
            kv_pool_reserved_bytes: ctx.kv_cache_pool().total_capacity() as u64,
            kv_pool_used_bytes: 0,
            forward_pass_breakdown: forward_pass_breakdown.clone(),
        });
    }

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, batch_size * n_heads, kv_head_dim)?;
    }

    if cache_shapes_changed {
        ctx.clear_cache();
    }

    // --- Prompt Processing Pass ---
    let mut logits_tensor: Option<Tensor<T>> = None;
    if !input_ids.is_empty() {
        ctx.with_gpu_scope("Prompt Processing", |ctx| {
            for (i, &token_id) in input_ids.iter().enumerate() {
                let input_tensor = qwen.embed(&[token_id], ctx)?;
                let (hidden_states, breakdown) = qwen.forward_step(&input_tensor, i, ctx)?;
                forward_pass_breakdown.extend(breakdown);
                logits_tensor = Some(ctx.with_gpu_scope("generation_step_output", |ctx| qwen.output(&hidden_states, ctx))?);
                log_cache_stats(ctx, "prompt", i + 1);
            }
            Ok::<(), MetalError>(())
        })?;
    }

    let mut generated_ids = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let vocab_size = qwen.config.vocab_size;
    let mut decoded_chunk = String::new();
    let mut decode_scratch = Vec::new();
    let base_seed = cfg.seed.unwrap_or_else(|| rand::rng().next_u32());

    let mut pending_output = if let Some(logits_tensor) = logits_tensor {
        let iteration_start = Instant::now();
        let mut initial_future = gpu_sample_top_k_top_p_async::<T>(
            &logits_tensor,
            vocab_size,
            cfg.top_k,
            cfg.top_p,
            cfg.temperature,
            Some(base_seed),
            ctx,
            iteration_start,
        )?;
        ctx.submit_active_command_buffer();
        initial_future.mark_submitted();
        ctx.poll_command_buffer_completions();
        let initial_sample = initial_future.wait(ctx)?;
        Some(CompletedToken::from_sample(initial_sample))
    } else {
        Some(CompletedToken {
            token_id: 0,
            sample_duration: Duration::ZERO,
            iteration_duration: Duration::ZERO,
            sample_wait_duration: Duration::ZERO,
        })
    };

    ctx.with_gpu_scope("Generation Loop", |ctx| {
        let mut metric_recording_overhead = Duration::ZERO;
        let mut last_memory_profiling = Instant::now();
        let memory_profiling_interval = Duration::from_millis(100);

        let mut current_input_token = pending_output.as_ref().map(|token| token.token_id).unwrap_or(0);

        for i in 0..cfg.max_tokens - 1 {
            if pending_output.is_none() {
                break;
            }

            let iteration_start = Instant::now();

            let reset_start = Instant::now();
            ctx.reset_pool();
            let reset_duration = reset_start.elapsed();
            if !reset_duration.is_zero() {
                let metric_start = Instant::now();
                record_metric_async!(MetricEvent::InternalKernelCompleted {
                    parent_op_name: "generation_loop".to_string(),
                    internal_kernel_name: "pool_reset".to_string(),
                    duration_us: reset_duration.as_micros() as u64,
                });
                metric_recording_overhead += metric_start.elapsed();
            }

            let current_pos = prompt_len + i;

            let embed_start = Instant::now();
            let input_tensor = embed_single_token_cached(qwen, current_input_token, ctx)?;
            let embed_duration = embed_start.elapsed();
            if !embed_duration.is_zero() {
                let metric_start = Instant::now();
                record_metric_async!(MetricEvent::InternalKernelCompleted {
                    parent_op_name: "generation_loop".to_string(),
                    internal_kernel_name: "embedding".to_string(),
                    duration_us: embed_duration.as_micros() as u64,
                });
                metric_recording_overhead += metric_start.elapsed();
            }

            let forward_step_start = Instant::now();
            let (hidden_states, breakdown) = qwen.forward_step(&input_tensor, current_pos, ctx)?;
            forward_pass_breakdown.extend(breakdown);
            let forward_step_duration = forward_step_start.elapsed();
            if !forward_step_duration.is_zero() {
                let metric_start = Instant::now();
                record_metric_async!(MetricEvent::InternalKernelCompleted {
                    parent_op_name: "generation_loop".to_string(),
                    internal_kernel_name: "forward_step_total".to_string(),
                    duration_us: forward_step_duration.as_micros() as u64,
                });
                metric_recording_overhead += metric_start.elapsed();
            }

            if let Ok(config) = metallic_instrumentation::config::AppConfig::get_or_init_from_env()
                && config.enable_profiling
                && last_memory_profiling.elapsed() >= memory_profiling_interval
            {
                let tensor_pool_used = ctx.pool.used_bytes();
                let kv_cache_used = kv_cache_size;
                let _total_memory_used = tensor_pool_used + kv_cache_used;

                let process_memory_bytes = global_cached_memory_profiler().get_process_memory_usage();
                last_memory_profiling = Instant::now();

                if process_memory_bytes > 1_048_576 || generated_ids.is_empty() {
                    record_metric_async!(MetricEvent::HostMemory {
                        total_bytes: process_memory_bytes,
                        tensor_pool_reserved_bytes: ctx.pool.total_capacity() as u64,
                        tensor_pool_used_bytes: ctx.pool.used_bytes() as u64,
                        kv_pool_reserved_bytes: ctx.kv_cache_pool().total_capacity() as u64,
                        kv_pool_used_bytes: ctx.kv_cache_pool().used_bytes() as u64,
                        forward_pass_breakdown: forward_pass_breakdown.clone(),
                    });
                }
            }

            let logits_tensor = ctx.with_gpu_scope("generation_step_output", |ctx| qwen.output(&hidden_states, ctx))?;

            if i == 0 {
                let bytes_per_element = T::DTYPE.size_bytes();
                let mut breakdown = FxHashMap::default();

                let embedding_size = (input_tensor.len() * bytes_per_element) as u64;
                breakdown.insert("Embedding".to_string(), embedding_size);

                let output_size = (logits_tensor.len() * bytes_per_element) as u64;
                breakdown.insert("Output".to_string(), output_size);

                let total_size = embedding_size + output_size;

                record_metric_async!(MetricEvent::ForwardStep {
                    total_bytes: total_size,
                    breakdown,
                });
            }

            let mut sample_future = gpu_sample_top_k_top_p_async::<T>(
                &logits_tensor,
                vocab_size,
                cfg.top_k,
                cfg.top_p,
                cfg.temperature,
                Some(base_seed.wrapping_add((i as u32).wrapping_add(1))),
                ctx,
                iteration_start,
            )?;
            ctx.submit_active_command_buffer();
            sample_future.mark_submitted();
            ctx.poll_command_buffer_completions();

            let completed_token = pending_output.take().expect("pending output token must exist before processing");
            let outcome = process_completed_token(
                completed_token,
                &mut generated_ids,
                tokenizer,
                &mut decoded_chunk,
                &mut decode_scratch,
                ctx,
                &mut metric_recording_overhead,
                token_callback,
            )?;
            if outcome.is_eos || !outcome.continue_generation {
                let _ = sample_future.wait(ctx)?;
                pending_output = None;
                break;
            }

            let sample_result = sample_future.wait(ctx)?;
            let next_completed = CompletedToken::from_sample(sample_result);
            current_input_token = next_completed.token_id;
            pending_output = Some(next_completed);
        }

        if let Some(remaining) = pending_output.take() {
            let _ = process_completed_token(
                remaining,
                &mut generated_ids,
                tokenizer,
                &mut decoded_chunk,
                &mut decode_scratch,
                ctx,
                &mut metric_recording_overhead,
                token_callback,
            )?;
        }

        if !metric_recording_overhead.is_zero() {
            record_metric_async!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "generation_loop".to_string(),
                internal_kernel_name: "metric_recording_overhead".to_string(),
                duration_us: metric_recording_overhead.as_micros() as u64,
            });
        }

        Ok::<(), MetalError>(())
    })?;
    Ok::<(), MetalError>(())
}

/// High-level end-to-end generation pipeline with token streaming support (Foundry backend)
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_foundry(
    foundry: &mut crate::foundry::Foundry,
    model: &crate::foundry::model::CompiledModel,
    prompt: &str,
    cfg: &GenerationConfig,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError> {
    // Build full prompt string following Qwen2.5 chat template
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );

    let generation_start = Instant::now();
    let prompt_start = Instant::now();
    let tokenizer = model.tokenizer()?;
    let prompt_tokens = tokenizer.encode(&full_prompt)?;
    let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);

    let mut prompt_processing_duration: Option<Duration> = None;

    // We need to maintain decode state (scratch buffer) to handle utf8 splitting
    let mut decode_scratch = Vec::new();
    let mut decoded_chunk = String::new();

    let callback = |token_id: u32| -> Result<bool, MetalError> {
        let now = Instant::now();
        // first token means prompt processing done (roughly)
        let prompt_duration = *prompt_processing_duration.get_or_insert_with(|| now.duration_since(prompt_start));

        // Decode
        // Foundry generates token IDs. Using Tokenizer::decode_token_arc logic
        if let Some(text) = tokenizer.decode_token_arc(token_id, &mut decoded_chunk, &mut decode_scratch)? {
            if tx
                .send(AppEvent::Token {
                    text,
                    prompt_processing: prompt_duration,
                    iteration: None, // Foundry doesn't report per-token time yet in callback
                })
                .is_err()
            {
                return Ok(false);
            }
        }
        Ok(true)
    };

    model.generate_with_seed_streaming(
        foundry,
        &prompt_tokens,
        cfg.max_tokens,
        &[eos],
        cfg.temperature,
        cfg.top_k as u32,
        cfg.top_p,
        cfg.seed.map(|s| s as u32).unwrap_or_else(|| rand::random()),
        callback,
    )?;

    // Send generation completion event with total generation time
    let total_generation_time = generation_start.elapsed();
    let _ = tx.send(AppEvent::GenerationComplete { total_generation_time });

    Ok(())
}
