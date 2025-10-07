use super::{Context, MetalError, SamplerBuffers, Tensor, resource_cache::CacheMetrics};
use crate::models::qwen25::Qwen25;
use crate::{TensorElement, Tokenizer};
use metallic_cli_helpers::app_event::AppEvent;
use metallic_instrumentation::{MetricEvent, record_metric};
use rand::prelude::*;
use std::{
    env,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

const IM_START: &str = "[:1]";
const IM_END: &str = "[:2]";

const METALLIC_LOG_CACHE_STATS_ENV: &str = "METALLIC_LOG_CACHE_STATS";
const METALLIC_LOG_CACHE_STATS_DEFAULT_FILE: &str = "metal-cache-stats.log";

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

    format!(
        "{name}(size={} hits={} misses={} requests={} hit_rate={hit_rate:.1}% last={last})",
        metrics.size, metrics.hits, metrics.misses, requests
    )
}

/// Generation configuration (defaults chosen by user)
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
        }
    }
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
    // Pre-allocate KV cache for all layers
    let n_layers = qwen.config.n_layers;
    let seq_len = qwen.config.seq_len;
    let n_kv_heads = qwen.config.n_kv_heads;
    let d_model = qwen.config.d_model;
    let n_heads = qwen.config.n_heads;
    let kv_dim = d_model * n_kv_heads / n_heads;
    let kv_head_dim = kv_dim / n_kv_heads;
    let batch_size = 1; // Assuming batch size of 1 for now
    let kv_capacity = (input_ids.len().max(1) + cfg.max_tokens).min(seq_len);

    // Determine whether existing cache-backed descriptors must be invalidated because
    // their shapes changed (e.g. when generating with a longer context than before).
    let repeated_batch_heads = batch_size * n_heads;
    let mut cache_shapes_changed = false;
    if !ctx.kv_caches.is_empty() {
        for layer_idx in 0..n_layers {
            match ctx.kv_caches.get(&layer_idx) {
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
    ctx.kv_cache_pool.reset();

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, batch_size * n_heads, kv_head_dim)?;
    }

    if cache_shapes_changed {
        ctx.clear_cache();
    }

    // --- Prompt Processing Pass ---
    // Process the prompt token by token to warm up the KV cache.
    let mut logits_tensor: Option<Tensor<T>> = None;
    if !input_ids.is_empty() {
        for (i, &token_id) in input_ids.iter().enumerate() {
            let input_tensor = qwen.embed(&[token_id], ctx)?;
            let hidden_states = qwen.forward_step(&input_tensor, i, ctx)?;
            // Record metrics using new system
            record_metric!(MetricEvent::GpuKernelDispatched {
                kernel_name: "forward_step".to_string(),
                op_name: format!("prompt_step_{}", i),
                thread_groups: (1, 1, 1),
            });
            logits_tensor = Some(ctx.with_gpu_scope(format!("prompt_step_{}", i), |ctx| qwen.output(&hidden_states, ctx))?);
            log_cache_stats(ctx, "prompt", i + 1);
        }
    }

    let mut generated_ids = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let vocab_size = qwen.config.vocab_size;
    let mut next_token;

    if let Some(logits_tensor) = logits_tensor {
        // Extract logits for the very last token of the prompt
        let logits = logits_tensor.to_vec();
        let vocab_logits = &logits[..vocab_size];

        // Sample the first token
        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature, &mut ctx.sampler_buffers) as u32;
        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "sampling".to_string(),
                internal_kernel_name: "top_k_top_p".to_string(),
                duration_us: sample_duration.as_micros() as u64,
            });
        }
    } else {
        // If there's no prompt, start with token 0.
        next_token = 0;
    }

    generated_ids.push(next_token);

    // --- Autoregressive Generation Loop ---
    // Now, generate tokens one by one using the KV cache.
    for i in 0..cfg.max_tokens - 1 {
        ctx.reset_pool();

        let current_pos = prompt_len + i;

        let embed_start = Instant::now();
        let input_tensor = qwen.embed(&[next_token], ctx)?;
        let embed_duration = embed_start.elapsed();
        if !embed_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "generation_loop".to_string(),
                internal_kernel_name: "embedding".to_string(),
                duration_us: embed_duration.as_micros() as u64,
            });
        }

        let hidden_states = ctx.with_gpu_scope(format!("generation_step_{}", i), |gpu_ctx| {
            qwen.forward_step(&input_tensor, current_pos, gpu_ctx)
        })?;
        record_metric!(MetricEvent::GpuKernelDispatched {
            kernel_name: "forward_step".to_string(),
            op_name: format!("generation_step_{}", i),
            thread_groups: (1, 1, 1),
        });

        let logits_tensor = ctx.with_gpu_scope(format!("generation_step_{}_output", i), |gpu_ctx| {
            qwen.output(&hidden_states, gpu_ctx)
        })?;

        let logits = logits_tensor.to_vec();
        let vocab_logits = &logits[..vocab_size];

        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature, &mut ctx.sampler_buffers) as u32;

        generated_ids.push(next_token);

        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "sampling".to_string(),
                internal_kernel_name: "top_k_top_p".to_string(),
                duration_us: sample_duration.as_micros() as u64,
            });
        }

        log_cache_stats(ctx, "generate", generated_ids.len());

        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        if next_token == eos_token_id {
            break;
        }
    }

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

    let prompt_start = Instant::now();

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    let mut prompt_processing_duration: Option<Duration> = None;
    let mut token_callback = |token_id, decoded_token: std::sync::Arc<str>, iteration_duration: Duration| -> Result<bool, MetalError> {
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
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError>
where
    F: FnMut(u32, std::sync::Arc<str>, Duration) -> Result<bool, MetalError>,
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
    let kv_capacity = (input_ids.len().max(1) + cfg.max_tokens).min(seq_len);

    // Determine whether existing cache-backed descriptors must be invalidated because
    // their shapes changed (e.g. when generating with a longer context than before).
    let repeated_batch_heads = batch_size * n_heads;
    let mut cache_shapes_changed = false;
    if !ctx.kv_caches.is_empty() {
        for layer_idx in 0..n_layers {
            match ctx.kv_caches.get(&layer_idx) {
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
    ctx.kv_cache_pool.reset();

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, batch_size * n_heads, kv_head_dim)?;
    }

    if cache_shapes_changed {
        ctx.clear_cache();
    }

    // --- Prompt Processing Pass ---
    // Process the prompt token by token to warm up the KV cache.
    let mut logits_tensor: Option<Tensor<T>> = None;
    if !input_ids.is_empty() {
        for (i, &token_id) in input_ids.iter().enumerate() {
            let input_tensor = qwen.embed(&[token_id], ctx)?;
            let hidden_states = qwen.forward_step(&input_tensor, i, ctx)?;
            // Record metrics using new system
            record_metric!(MetricEvent::GpuKernelDispatched {
                kernel_name: "forward_step".to_string(),
                op_name: format!("prompt_step_{}", i),
                thread_groups: (1, 1, 1),
            });
            logits_tensor = Some(ctx.with_gpu_scope(format!("prompt_step_{}", i), |ctx| qwen.output(&hidden_states, ctx))?);
            log_cache_stats(ctx, "prompt", i + 1);
        }
    }

    let mut generated_ids = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let vocab_size = qwen.config.vocab_size;
    let mut next_token;
    let mut decoded_chunk = String::new();
    let mut decode_scratch = Vec::new();

    if let Some(logits_tensor) = logits_tensor {
        // Extract logits for the very last token of the prompt
        let logits = logits_tensor.to_vec();
        let vocab_logits = &logits[..vocab_size];

        // Sample the first token
        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature, &mut ctx.sampler_buffers) as u32;
        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "sampling".to_string(),
                internal_kernel_name: "top_k_top_p".to_string(),
                duration_us: sample_duration.as_micros() as u64,
            });
        }
    } else {
        // If there's no prompt, start with token 0.
        next_token = 0;
    }

    generated_ids.push(next_token);
    let decode_start = Instant::now();
    let decoded_piece = tokenizer.decode_token_arc(next_token, &mut decoded_chunk, &mut decode_scratch)?;
    let decode_duration = decode_start.elapsed();
    if !decode_duration.is_zero() {
        record_metric!(MetricEvent::InternalKernelCompleted {
            parent_op_name: "decoding".to_string(),
            internal_kernel_name: "token_decode".to_string(),
            duration_us: decode_duration.as_micros() as u64,
        });
    }

    log_cache_stats(ctx, "generate", generated_ids.len());

    if let Some(piece) = decoded_piece
        && !token_callback(next_token, piece, Duration::ZERO)?
    {
        return Ok(());
    }

    // --- Autoregressive Generation Loop ---
    // Now, generate tokens one by one using the KV cache.
    let mut ui_connected = true;

    for i in 0..cfg.max_tokens - 1 {
        let iteration_start = Instant::now();
        ctx.reset_pool();

        let current_pos = prompt_len + i;

        let embed_start = Instant::now();
        let input_tensor = qwen.embed(&[next_token], ctx)?;
        let embed_duration = embed_start.elapsed();
        if !embed_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "generation_loop".to_string(),
                internal_kernel_name: "embedding".to_string(),
                duration_us: embed_duration.as_micros() as u64,
            });
        }

        let hidden_states = qwen.forward_step(&input_tensor, current_pos, ctx)?;
        record_metric!(MetricEvent::GpuKernelDispatched {
            kernel_name: "forward_step".to_string(),
            op_name: format!("generation_step_{}", i),
            thread_groups: (1, 1, 1),
        });

        let output_start = Instant::now();
        let logits_tensor = ctx.with_gpu_scope(format!("generation_step_{}_output", i), |ctx| qwen.output(&hidden_states, ctx))?;
        let output_duration = output_start.elapsed();
        if !output_duration.is_zero() {
            record_metric!(MetricEvent::GpuOpCompleted {
                op_name: format!("generation_step_{}_output", i),
                backend: "Metal".to_string(),
                duration_us: output_duration.as_micros() as u64,
            });
        }

        let logits = logits_tensor.to_vec();
        let vocab_logits = &logits[..vocab_size];

        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature, &mut ctx.sampler_buffers) as u32;

        generated_ids.push(next_token);

        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "sampling".to_string(),
                internal_kernel_name: "top_k_top_p".to_string(),
                duration_us: sample_duration.as_micros() as u64,
            });
        }

        let decode_start = Instant::now();
        let decoded_piece = tokenizer.decode_token_arc(next_token, &mut decoded_chunk, &mut decode_scratch)?;
        let decode_duration = decode_start.elapsed();
        if !decode_duration.is_zero() {
            record_metric!(MetricEvent::InternalKernelCompleted {
                parent_op_name: "decoding".to_string(),
                internal_kernel_name: "token_decode".to_string(),
                duration_us: decode_duration.as_micros() as u64,
            });
        }

        let iteration_duration = iteration_start.elapsed();
        if !iteration_duration.is_zero() {
            record_metric!(MetricEvent::GpuOpCompleted {
                op_name: format!("iteration_{}", i),
                backend: "Metal".to_string(),
                duration_us: iteration_duration.as_micros() as u64,
            });
        }

        log_cache_stats(ctx, "generate", generated_ids.len());

        if let Some(piece) = decoded_piece
            && !token_callback(next_token, piece, iteration_duration)?
        {
            break;
        }

        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        if next_token == eos_token_id {
            break;
        }
    }

    Ok(())
}
