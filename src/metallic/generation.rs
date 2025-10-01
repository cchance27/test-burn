use super::{Context, MetalError, Tensor};
use crate::metallic::instrumentation::{MemoryEvent, MemoryUsage, new_latency_collector, new_memory_collector};
use crate::metallic::metrics::{
    BlockStat, MemoryBlockStat, MemoryScopeStat, MetricsLoggers, ModelMemoryNode, ProcessMemoryTracker, RollingStat, ScalarStat,
    SoftmaxBackendStats, build_latency_rows, build_memory_rows, build_model_memory_tree, log_interval_from_env, sample_process_memory,
};
use crate::metallic::models::qwen25::Qwen25;
use crate::metallic::{TensorElement, Tokenizer};
use crate::{alert, app_event::AppEvent};
use rand::prelude::*;
use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

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
pub fn sample_top_k_top_p<T: TensorElement>(logits: &[T::Scalar], top_k: usize, top_p: f32, temperature: f32) -> usize {
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

    // Apply temperature scaling and convert to positive scores
    let mut scaled: Vec<f32> = logits.iter().map(|&v| T::to_f32(v) / temperature).collect();

    // Stabilize by subtracting max before exponentiation to prevent overflow
    // Filter out any infinity/nan values first
    let finite_scaled: Vec<f32> = scaled.iter().cloned().filter(|x| x.is_finite()).collect();
    if finite_scaled.is_empty() {
        return if fallback_found { fallback_idx } else { 0 };
    }

    let m = finite_scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Apply the shift and compute exponentials
    for x in &mut scaled {
        if x.is_finite() {
            *x = (*x - m).exp();
            // Clamp extremely large values to prevent overflow
            if *x > 1e10 {
                *x = 1e10;
            }
            // Clamp extremely small values to prevent underflow
            if *x < 1e-10 {
                *x = 0.0;
            }
        } else {
            *x = 0.0; // Replace non-finite values with 0
        }
    }

    // Normalize to probabilities
    let sum: f32 = scaled.iter().sum();
    if sum <= 0.0 || sum.is_infinite() || sum.is_nan() {
        return if fallback_found { fallback_idx } else { 0 };
    }
    for x in &mut scaled {
        *x /= sum;
    }

    // Sort indices by probability descending
    let mut idxs: Vec<usize> = (0..scaled.len()).collect();
    idxs.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top-k filtering first
    let k_cutoff = std::cmp::min(top_k, idxs.len());
    let idxs = &idxs[0..k_cutoff];

    // Then apply top-p filtering
    let mut cum = 0.0f32;
    let mut cutoff = 0usize;
    for (i, &id) in idxs.iter().enumerate() {
        cum += scaled[id];
        cutoff = i;
        if cum >= top_p || cum.is_infinite() || cum.is_nan() {
            break;
        }
    }

    let shortlist = &idxs[0..=cutoff];
    let mut shortlist_probs: Vec<f32> = shortlist.iter().map(|&i| scaled[i]).collect();
    let ssum: f32 = shortlist_probs.iter().sum();
    if ssum <= 0.0 || ssum.is_infinite() || ssum.is_nan() {
        return shortlist.first().copied().unwrap_or(if fallback_found { fallback_idx } else { 0 });
    }
    for p in &mut shortlist_probs {
        *p /= ssum;
    }

    // Sample using RNG (use simple rng.next_u32() -> float to avoid trait issues)
    let mut rng = rand::rng();
    let r = (rng.next_u32() as f32) / (u32::MAX as f32);
    let mut acc = 0.0f32;
    for (i, &p) in shortlist_probs.iter().enumerate() {
        acc += p;
        if r <= acc || acc.is_infinite() || acc.is_nan() {
            return shortlist[i];
        }
    }
    shortlist[shortlist.len() - 1]
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
    let output_tokens = generate_autoregressive_with_kv_cache(qwen, tokenizer, ctx, &input_ids, cfg, &[])?;
    let output_text = tokenizer.decode(&output_tokens)?;
    Ok(output_text)
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
    host_overheads: &[(String, usize)],
    host_memory: &mut ScalarStat,
    process_memory_tracker: &mut Option<ProcessMemoryTracker>,
) -> Result<(), MetalError> {
    // Build full prompt string following Qwen2.5 chat template
    let full_prompt = format!(
        "{IM_START}\nsystem\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.{IM_END}\n{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
    );

    let prompt_start = Instant::now();

    // Encode the full prompt
    let input_ids = tokenizer.encode(&full_prompt)?;

    let mut token_count = 0usize;
    let mut prompt_processing_duration: Option<Duration> = None;
    let mut generation_start: Option<Instant> = None;

    let mut token_callback = |_token_id, decoded_token: Arc<str>| -> Result<bool, MetalError> {
        token_count += 1;
        let now = Instant::now();

        let prompt_duration = *prompt_processing_duration.get_or_insert_with(|| now.duration_since(prompt_start));

        let gen_start = generation_start.get_or_insert(now);
        let generation_elapsed = now.duration_since(*gen_start);
        let elapsed_secs = generation_elapsed.as_secs_f64();
        let tokens_per_second = if elapsed_secs > 0.0 {
            token_count as f64 / elapsed_secs
        } else {
            0.0
        };

        if tx
            .send(AppEvent::Token {
                text: decoded_token,
                tokens_per_second,
                prompt_processing: prompt_duration,
                generation: generation_elapsed,
            })
            .is_err()
        {
            return Ok(false); // Stop generation if UI thread has disconnected
        }
        Ok(true)
    };

    // Generate tokens using the new KV cache approach
    generate_autoregressive_with_kv_cache_streaming(
        qwen,
        tokenizer,
        ctx,
        &input_ids,
        cfg,
        &mut token_callback,
        tx,
        host_overheads,
        host_memory,
        process_memory_tracker,
    )?;

    Ok(())
}

/// High-level autoregressive generation loop using Qwen25 with KV caches for debugging.
/// This implementation processes the full context each time for comparison.
pub fn generate_autoregressive_with_kv_cache<T: TensorElement>(
    qwen: &mut Qwen25<T>,
    tokenizer: &Tokenizer,
    ctx: &mut Context<T>,
    input_ids: &[u32],
    cfg: &GenerationConfig,
    host_overheads: &[(String, usize)],
) -> Result<Vec<u32>, MetalError> {
    let mut result = Vec::new();
    let mut callback = |token_id, _decoded_token: Arc<str>| -> Result<bool, MetalError> {
        result.push(token_id);
        Ok(true)
    };

    let (tx, _) = mpsc::channel();
    let mut host_memory = ScalarStat::default();
    let mut process_memory_tracker = ProcessMemoryTracker::new();
    generate_autoregressive_with_kv_cache_streaming(
        qwen,
        tokenizer,
        ctx,
        input_ids,
        cfg,
        &mut callback,
        &tx,
        host_overheads,
        &mut host_memory,
        &mut process_memory_tracker,
    )?;

    Ok(result)
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
    host_overheads: &[(String, usize)],
    host_memory: &mut ScalarStat,
    process_memory_tracker: &mut Option<ProcessMemoryTracker>,
) -> Result<(), MetalError>
where
    F: FnMut(u32, Arc<str>) -> Result<bool, MetalError>,
{
    // Ensure KV caches start from a clean slate between generations.
    ctx.kv_caches.clear();
    ctx.kv_cache_pool.reset();

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

    let log_interval = log_interval_from_env();
    let mut metrics_loggers = MetricsLoggers::from_env(log_interval);

    let mut embed_stats = RollingStat::default();
    let mut forward_stats = RollingStat::default();
    let mut output_stats = RollingStat::default();
    let mut sample_stats = RollingStat::default();
    let mut decode_stats = RollingStat::default();
    let mut block_stats = vec![BlockStat::default(); n_layers];
    let mut softmax_backend_stats = SoftmaxBackendStats::default();
    let mut latencies_ready = false;
    let mut memory_embed = MemoryScopeStat::default();
    let mut memory_forward = MemoryScopeStat::default();
    let mut memory_output = MemoryScopeStat::default();
    let mut memory_blocks = vec![MemoryBlockStat::default(); n_layers];
    let mut memory_ready = false;
    sample_process_memory(process_memory_tracker, host_memory);
    let model_memory_tree = build_model_memory_tree(qwen);

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, kv_capacity, batch_size * n_kv_heads, batch_size * n_heads, kv_head_dim)?;
    }

    let mut latest_forward_usage = Some(ctx.snapshot_memory_usage());
    sample_process_memory(process_memory_tracker, host_memory);

    // --- Prompt Processing Pass ---
    // Process the prompt token by token to warm up the KV cache.
    let mut logits_tensor: Option<Tensor<T>> = None;
    if !input_ids.is_empty() {
        ctx.clear_cache(); // It's okay to clear the resource cache
        for (i, &token_id) in input_ids.iter().enumerate() {
            let input_tensor = qwen.embed(&[token_id], ctx)?;
            let hidden_states = qwen.forward_step(&input_tensor, i, ctx)?;
            for sample in ctx.take_softmax_samples() {
                softmax_backend_stats.record(sample.backend, sample.duration);
            }
            logits_tensor = Some(qwen.output(&hidden_states, ctx)?);
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
        let vocab_logits = logits[0..vocab_size].to_vec();

        // Sample the first token
        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;
        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            sample_stats.record(sample_duration);
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
        decode_stats.record(decode_duration);
    }

    if let Some(piece) = decoded_piece
        && !token_callback(next_token, piece)?
    {
        return Ok(());
    }

    // --- Autoregressive Generation Loop ---
    // Now, generate tokens one by one using the KV cache.
    let mut ui_connected = true;

    emit_memory_rows(
        &model_memory_tree,
        host_memory,
        &memory_embed,
        &memory_forward,
        latest_forward_usage,
        &memory_blocks,
        &memory_output,
        host_overheads,
        &mut metrics_loggers,
        tx,
        &mut ui_connected,
        true,
    );
    for i in 0..cfg.max_tokens - 1 {
        ctx.reset_pool();
        ctx.clear_cache();

        let embed_usage_before = ctx.snapshot_memory_usage();
        let embed_start = Instant::now();
        let input_tensor = qwen.embed(&[next_token], ctx)?;
        let embed_duration = embed_start.elapsed();
        if !embed_duration.is_zero() {
            embed_stats.record(embed_duration);
        }
        let embed_usage_after = ctx.snapshot_memory_usage();
        let embed_delta = embed_usage_after.delta_from(embed_usage_before);
        memory_embed.update(
            embed_delta.pool_used,
            embed_delta.kv_used,
            embed_delta.kv_cache_bytes,
            embed_delta.pool_used,
            embed_delta.kv_used,
            embed_delta.kv_cache_bytes,
        );

        let current_pos = prompt_len + i;
        let latency_collector = new_latency_collector(n_layers);
        let memory_collector = new_memory_collector(n_layers);
        ctx.set_latency_collector(Some(latency_collector.clone()));
        ctx.set_memory_collector(Some(memory_collector.clone()));
        ctx.record_memory_event(MemoryEvent::ForwardStart);

        let hidden_states = qwen.forward_step(&input_tensor, current_pos, ctx)?;
        let forward_snapshot = latency_collector.borrow().snapshot();
        let memory_snapshot = memory_collector.borrow().snapshot();
        ctx.set_latency_collector(None);
        ctx.set_memory_collector(None);

        for sample in ctx.take_softmax_samples() {
            softmax_backend_stats.record(sample.backend, sample.duration);
        }

        if let Some(usage) = memory_snapshot.forward.last {
            latest_forward_usage = Some(usage);
        }
        if memory_snapshot.forward.baseline.is_some() {
            memory_forward.update(
                memory_snapshot.forward.current_pool_delta,
                memory_snapshot.forward.current_kv_delta,
                memory_snapshot.forward.current_kv_cache_delta,
                memory_snapshot.forward.peak_pool_delta,
                memory_snapshot.forward.peak_kv_delta,
                memory_snapshot.forward.peak_kv_cache_delta,
            );
            for (idx, block_snapshot) in memory_snapshot.blocks.iter().enumerate() {
                memory_blocks[idx].update_from_snapshot(block_snapshot);
            }
            memory_ready = true;
        }

        if !forward_snapshot.forward_step.is_zero() {
            forward_stats.record(forward_snapshot.forward_step);
            latencies_ready = true;
        }
        for (idx, block_snapshot) in forward_snapshot.blocks.iter().enumerate() {
            if !block_snapshot.total.is_zero() {
                block_stats[idx].record_total(block_snapshot.total);
            }
            for phase in &block_snapshot.phases {
                if !phase.duration.is_zero() {
                    block_stats[idx].record_phase(&phase.label, phase.duration);
                }
            }
        }

        let output_usage_before = ctx.snapshot_memory_usage();
        let output_start = Instant::now();
        let logits_tensor = qwen.output(&hidden_states, ctx)?;
        let output_duration = output_start.elapsed();
        if !output_duration.is_zero() {
            output_stats.record(output_duration);
        }
        let output_usage_after = ctx.snapshot_memory_usage();
        let output_delta = output_usage_after.delta_from(output_usage_before);
        memory_output.update(
            output_delta.pool_used,
            output_delta.kv_used,
            output_delta.kv_cache_bytes,
            output_delta.pool_used,
            output_delta.kv_used,
            output_delta.kv_cache_bytes,
        );

        sample_process_memory(process_memory_tracker, host_memory);

        let logits = logits_tensor.to_vec();
        let vocab_logits = logits[0..vocab_size].to_vec();

        let sample_start = Instant::now();
        next_token = sample_top_k_top_p::<T>(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;

        generated_ids.push(next_token);

        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            sample_stats.record(sample_duration);
        }

        let decode_start = Instant::now();
        let decoded_piece = tokenizer.decode_token_arc(next_token, &mut decoded_chunk, &mut decode_scratch)?;
        let decode_duration = decode_start.elapsed();
        if !decode_duration.is_zero() {
            decode_stats.record(decode_duration);
        }

        if let Some(piece) = decoded_piece
            && !token_callback(next_token, piece)?
        {
            break;
        }

        if latencies_ready {
            let rows = build_latency_rows(
                &embed_stats,
                &forward_stats,
                &block_stats,
                &softmax_backend_stats,
                &output_stats,
                &sample_stats,
                &decode_stats,
            );
            if let Some(loggers) = metrics_loggers.as_mut() {
                let log_now = Instant::now();
                if let Err(err) = loggers.log_latency(&rows, log_now, false) {
                    alert::emit_warning(tx, format!("Failed to log latency metrics: {err}"));
                }
            }
            if ui_connected && tx.send(AppEvent::LatencyUpdate(rows)).is_err() {
                ui_connected = false;
            }
        }

        if memory_ready {
            emit_memory_rows(
                &model_memory_tree,
                host_memory,
                &memory_embed,
                &memory_forward,
                latest_forward_usage,
                &memory_blocks,
                &memory_output,
                host_overheads,
                &mut metrics_loggers,
                tx,
                &mut ui_connected,
                false,
            );
        }

        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        if next_token == eos_token_id {
            break;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_memory_rows(
    model_memory_tree: &ModelMemoryNode,
    host_memory: &ScalarStat,
    memory_embed: &MemoryScopeStat,
    memory_forward: &MemoryScopeStat,
    latest_forward_usage: Option<MemoryUsage>,
    memory_blocks: &[MemoryBlockStat],
    memory_output: &MemoryScopeStat,
    host_overheads: &[(String, usize)],
    metrics_loggers: &mut Option<MetricsLoggers>,
    tx: &mpsc::Sender<AppEvent>,
    ui_connected: &mut bool,
    force: bool,
) {
    let rows = build_memory_rows(
        model_memory_tree,
        host_memory,
        memory_embed,
        memory_forward,
        latest_forward_usage,
        memory_blocks,
        memory_output,
        host_overheads,
    );

    if let Some(loggers) = metrics_loggers.as_mut() {
        let log_now = Instant::now();
        if let Err(err) = loggers.log_memory(&rows, log_now, force) {
            alert::emit_warning(tx, format!("Failed to log memory metrics: {err}"));
        }
    }

    if *ui_connected && tx.send(AppEvent::MemoryUpdate(rows)).is_err() {
        *ui_connected = false;
    }
}
