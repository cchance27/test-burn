use super::{Context, MetalError, Tensor};
use crate::app_event::{AppEvent, LatencyRow};
use crate::metallic::instrumentation::new_collector;
use crate::metallic::models::qwen25::Qwen25;
use crate::metallic::Tokenizer;
use app_memory_usage_fetcher::get_memory_usage_mbytes;
use rand::prelude::*;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

#[derive(Clone, Copy, Default)]
struct RollingStat {
    last: Duration,
    total: Duration,
    count: u64,
}

impl RollingStat {
    fn record(&mut self, duration: Duration) {
        self.last = duration;
        self.total += duration;
        self.count += 1;
    }

    fn last_ms(&self) -> f64 {
        self.last.as_secs_f64() * 1000.0
    }

    fn average_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.total.as_secs_f64() * 1000.0) / self.count as f64
        }
    }
}

#[derive(Clone, Default)]
struct BlockPhaseStat {
    label: String,
    stat: RollingStat,
}

#[derive(Clone, Default)]
struct BlockStat {
    total: RollingStat,
    phases: Vec<BlockPhaseStat>,
}

impl BlockStat {
    fn record_total(&mut self, duration: Duration) {
        self.total.record(duration);
    }

    fn record_phase(&mut self, label: &str, duration: Duration) {
        if let Some(entry) = self.phases.iter_mut().find(|entry| entry.label == label) {
            entry.stat.record(duration);
        } else {
            let mut stat = RollingStat::default();
            stat.record(duration);
            self.phases.push(BlockPhaseStat {
                label: label.to_string(),
                stat,
            });
        }
    }
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
pub fn sample_top_k_top_p(logits: &[f32], top_k: usize, top_p: f32, temperature: f32) -> usize {
    // Handle deterministic (greedy) sampling when temperature is zero or non-finite.
    if temperature <= 0.0 || !temperature.is_finite() {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Apply temperature scaling and convert to positive scores
    let mut scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    // Stabilize by subtracting max before exponentiation to prevent overflow
    // Filter out any infinity/nan values first
    let finite_scaled: Vec<f32> = scaled.iter().cloned().filter(|x| x.is_finite()).collect();
    if finite_scaled.is_empty() {
        return 0; // fallback if all logits are non-finite
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
        return 0usize; // fallback
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
        return shortlist[0];
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
pub fn generate(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
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

/// High-level end-to-end generation pipeline with token streaming support
pub fn generate_streaming(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
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

    let mut token_count = 0usize;
    let mut prompt_processing_duration: Option<Duration> = None;
    let mut generation_start: Option<Instant> = None;

    let mut token_callback = |_token_id, decoded_token: String| -> Result<bool, MetalError> {
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
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, &input_ids, cfg, &mut token_callback, tx)?;

    Ok(())
}

/// High-level autoregressive generation loop using Qwen25 with KV caches for debugging.
/// This implementation processes the full context each time for comparison.
pub fn generate_autoregressive_with_kv_cache(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    input_ids: &[u32],
    cfg: &GenerationConfig,
) -> Result<Vec<u32>, MetalError> {
    let mut result = Vec::new();
    let mut callback = |token_id, _decoded_token| -> Result<bool, MetalError> {
        result.push(token_id);
        Ok(true)
    };

    let (tx, _) = mpsc::channel();
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, input_ids, cfg, &mut callback, &tx)?;

    Ok(result)
}

/// High-level autoregressive generation loop with streaming support using Qwen25 with KV Caching.
pub fn generate_autoregressive_with_kv_cache_streaming<F>(
    qwen: &mut Qwen25,
    tokenizer: &Tokenizer,
    ctx: &mut Context,
    input_ids: &[u32],
    cfg: &GenerationConfig,
    token_callback: &mut F,
    tx: &mpsc::Sender<AppEvent>,
) -> Result<(), MetalError>
where
    F: FnMut(u32, String) -> Result<bool, MetalError>,
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

    let mut embed_stats = RollingStat::default();
    let mut forward_stats = RollingStat::default();
    let mut output_stats = RollingStat::default();
    let mut sample_stats = RollingStat::default();
    let mut block_stats = vec![BlockStat::default(); n_layers];
    let mut latencies_ready = false;

    for layer_idx in 0..n_layers {
        ctx.alloc_kv_cache(layer_idx, seq_len, batch_size * n_kv_heads, kv_head_dim)?;
    }

    // --- Prompt Processing Pass ---
    // Process the prompt token by token to warm up the KV cache.
    let mut logits_tensor: Option<Tensor> = None;
    if !input_ids.is_empty() {
        ctx.clear_cache(); // It's okay to clear the resource cache
        for (i, &token_id) in input_ids.iter().enumerate() {
            let input_tensor = qwen.embed(&[token_id], ctx)?;
            let hidden_states = qwen.forward_step(&input_tensor, i, ctx)?;
            logits_tensor = Some(qwen.output(&hidden_states, ctx)?);
        }
    }

    let mut generated_ids = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let vocab_size = qwen.config.vocab_size;
    let mut next_token;
    let mut last_decoded_len = 0usize;

    if let Some(logits_tensor) = logits_tensor {
        // Extract logits for the very last token of the prompt
        let logits = logits_tensor.to_vec();
        let vocab_logits = logits[0..vocab_size].to_vec();

        // Sample the first token
        next_token = sample_top_k_top_p(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;
    } else {
        // If there's no prompt, start with token 0.
        next_token = 0;
    }

    generated_ids.push(next_token);
    let decoded_full = tokenizer.decode_lossless(&generated_ids[prompt_len..])?;
    let mut decoded_chunk = String::new();
    if decoded_full.len() >= last_decoded_len {
        decoded_chunk.push_str(&decoded_full[last_decoded_len..]);
    } else {
        decoded_chunk = decoded_full.clone();
    }
    last_decoded_len = decoded_full.len();

    if !token_callback(next_token, decoded_chunk)? {
        return Ok(());
    }

    // --- Autoregressive Generation Loop ---
    // Now, generate tokens one by one using the KV cache.
    let mut ui_connected = true;
    for i in 0..cfg.max_tokens - 1 {
        ctx.reset_pool();
        ctx.clear_cache();

        let embed_start = Instant::now();
        let input_tensor = qwen.embed(&[next_token], ctx)?;
        let embed_duration = embed_start.elapsed();
        if !embed_duration.is_zero() {
            embed_stats.record(embed_duration);
        }

        let current_pos = prompt_len + i;
        let collector = new_collector(n_layers);
        ctx.set_latency_collector(Some(collector.clone()));

        let hidden_states = qwen.forward_step(&input_tensor, current_pos, ctx)?;
        let forward_snapshot = collector.borrow().snapshot();
        ctx.set_latency_collector(None);

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

        let output_start = Instant::now();
        let logits_tensor = qwen.output(&hidden_states, ctx)?;
        let output_duration = output_start.elapsed();
        if !output_duration.is_zero() {
            output_stats.record(output_duration);
        }

        if ui_connected && let Some(stats) = ctx.get_cache_stats() {
            let app_mem = get_memory_usage_mbytes(); // This actually seems to return GB not MB due to a bug in crate
            let app_mem_str = app_mem.map(|mb| format!("{:.2} GB", mb)).unwrap_or_else(|| "n/a".to_string());

            let pool_used_mb = ctx.pool.used_bytes() as f32 / 1024.0 / 1024.0;
            let pool_capacity_mb = ctx.pool.total_capacity() as f32 / 1024.0 / 1024.0;

            let kv_used_mb = ctx.kv_cache_pool.used_bytes() as f32 / 1024.0 / 1024.0;
            let kv_capacity_mb = ctx.kv_cache_pool.total_capacity() as f32 / 1024.0 / 1024.0;
            let kv_layers = ctx.kv_caches.len();

            let memory_usage = format!(
                "App: {}\nPool: {:.2}/{:.2} MB (chunks: {}, allocs: {}, resets: {})\nKV: {:.2}/{:.2} MB (chunks: {}, allocs: {}, resets: {})\nContext: step {}/{} | Cache: G{}/D{}/S{}",
                app_mem_str,
                pool_used_mb,
                pool_capacity_mb,
                ctx.pool.num_chunks(),
                ctx.pool.pooled_allocations,
                ctx.pool.pool_resets,
                kv_used_mb,
                kv_capacity_mb,
                ctx.kv_cache_pool.num_chunks(),
                ctx.kv_cache_pool.pooled_allocations,
                ctx.kv_cache_pool.pool_resets,
                current_pos + 1,
                seq_len,
                stats.gemm_cache_size,
                stats.descriptor_cache_size,
                stats.sdpa_cache_size
            );

            if tx.send(AppEvent::MemoryUpdate(memory_usage)).is_err() {
                ui_connected = false;
            }
        }

        let logits = logits_tensor.to_vec();
        let vocab_logits = logits[0..vocab_size].to_vec();

        let sample_start = Instant::now();
        next_token = sample_top_k_top_p(&vocab_logits, cfg.top_k, cfg.top_p, cfg.temperature) as u32;

        generated_ids.push(next_token);

        let decoded_full = tokenizer.decode_lossless(&generated_ids[prompt_len..])?;
        let mut decoded_chunk = String::new();
        if decoded_full.len() >= last_decoded_len {
            decoded_chunk.push_str(&decoded_full[last_decoded_len..]);
        } else {
            decoded_chunk = decoded_full.clone();
        }
        last_decoded_len = decoded_full.len();

        let sample_duration = sample_start.elapsed();
        if !sample_duration.is_zero() {
            sample_stats.record(sample_duration);
        }

        if latencies_ready && ui_connected {
            let rows = build_latency_rows(&embed_stats, &forward_stats, &block_stats, &output_stats, &sample_stats);
            if tx.send(AppEvent::LatencyUpdate(rows)).is_err() {
                ui_connected = false;
            }
        }

        if !token_callback(next_token, decoded_chunk)? {
            break;
        }
        let eos_token_id = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        if next_token == eos_token_id {
            break;
        }
    }

    Ok(())
}

fn build_latency_rows(
    embed: &RollingStat,
    forward: &RollingStat,
    blocks: &[BlockStat],
    output: &RollingStat,
    sample: &RollingStat,
) -> Vec<LatencyRow> {
    let mut rows = Vec::new();

    rows.push(LatencyRow {
        label: "Embedding".to_string(),
        last_ms: embed.last_ms(),
        average_ms: embed.average_ms(),
        level: 0,
    });

    rows.push(LatencyRow {
        label: "Forward Step".to_string(),
        last_ms: forward.last_ms(),
        average_ms: forward.average_ms(),
        level: 0,
    });

    for (idx, stat) in blocks.iter().enumerate() {
        rows.push(LatencyRow {
            label: format!("Block {}", idx + 1),
            last_ms: stat.total.last_ms(),
            average_ms: stat.total.average_ms(),
            level: 1,
        });
        for phase in &stat.phases {
            rows.push(LatencyRow {
                label: phase.label.clone(),
                last_ms: phase.stat.last_ms(),
                average_ms: phase.stat.average_ms(),
                level: 2,
            });
        }
    }

    rows.push(LatencyRow {
        label: "Output".to_string(),
        last_ms: output.last_ms(),
        average_ms: output.average_ms(),
        level: 0,
    });

    rows.push(LatencyRow {
        label: "Sampling".to_string(),
        last_ms: sample.last_ms(),
        average_ms: sample.average_ms(),
        level: 0,
    });

    rows
}
