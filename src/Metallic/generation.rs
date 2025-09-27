use super::{Context, MetalError, Tensor};
use crate::app_event::{AppEvent, LatencyRow, MemoryRow};
use crate::metallic::instrumentation::{new_latency_collector, new_memory_collector, BlockMemorySnapshot, MemoryEvent, MemoryUsage};
use crate::metallic::models::qwen25::Qwen25;
use crate::metallic::Tokenizer;
use rand::prelude::*;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use sysinfo::{get_current_pid, Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

#[derive(Clone)]
struct ModelMemoryNode {
    label: String,
    bytes: usize,
    children: Vec<ModelMemoryNode>,
}

impl ModelMemoryNode {
    fn leaf(label: impl Into<String>, bytes: usize) -> Self {
        Self {
            label: label.into(),
            bytes,
            children: Vec::new(),
        }
    }

    fn branch(label: impl Into<String>, children: Vec<ModelMemoryNode>) -> Self {
        Self {
            label: label.into(),
            bytes: 0,
            children,
        }
    }

    fn total_bytes(&self) -> usize {
        self.bytes + self.children.iter().map(|child| child.total_bytes()).sum::<usize>()
    }
}

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

#[derive(Clone, Default)]
struct MemoryScopeStat {
    current_pool_mb: f64,
    peak_pool_mb: f64,
    current_kv_mb: f64,
    peak_kv_mb: f64,
    current_kv_cache_mb: f64,
    peak_kv_cache_mb: f64,
}

impl MemoryScopeStat {
    fn update(
        &mut self,
        current_pool_bytes: usize,
        current_kv_bytes: usize,
        current_kv_cache_bytes: usize,
        peak_pool_bytes: usize,
        peak_kv_bytes: usize,
        peak_kv_cache_bytes: usize,
    ) {
        self.current_pool_mb = bytes_to_mb(current_pool_bytes);
        self.current_kv_mb = bytes_to_mb(current_kv_bytes);
        self.current_kv_cache_mb = bytes_to_mb(current_kv_cache_bytes);
        self.peak_pool_mb = self.peak_pool_mb.max(bytes_to_mb(peak_pool_bytes));
        self.peak_kv_mb = self.peak_kv_mb.max(bytes_to_mb(peak_kv_bytes));
        self.peak_kv_cache_mb = self.peak_kv_cache_mb.max(bytes_to_mb(peak_kv_cache_bytes));
    }

    fn has_data(&self) -> bool {
        self.current_total_mb() > 0.0 || self.peak_total_mb() > 0.0
    }

    fn current_total_mb(&self) -> f64 {
        self.current_pool_mb + self.current_kv_mb
    }

    fn peak_total_mb(&self) -> f64 {
        self.peak_pool_mb + self.peak_kv_mb
    }

    fn to_row(&self, label: String, level: u8) -> MemoryRow {
        MemoryRow {
            label,
            level,
            current_total_mb: self.current_total_mb(),
            peak_total_mb: self.peak_total_mb(),
            current_pool_mb: self.current_pool_mb,
            peak_pool_mb: self.peak_pool_mb,
            current_kv_mb: self.current_kv_mb,
            peak_kv_mb: self.peak_kv_mb,
            current_kv_cache_mb: self.current_kv_cache_mb,
            peak_kv_cache_mb: self.peak_kv_cache_mb,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        }
    }
}

#[derive(Clone, Default)]
struct MemoryPhaseStat {
    label: String,
    scope: MemoryScopeStat,
}

#[derive(Clone, Default)]
struct MemoryBlockStat {
    scope: MemoryScopeStat,
    phases: Vec<MemoryPhaseStat>,
}

impl MemoryBlockStat {
    fn update_from_snapshot(&mut self, snapshot: &BlockMemorySnapshot) {
        self.scope.update(
            snapshot.current_pool_delta,
            snapshot.current_kv_delta,
            snapshot.current_kv_cache_delta,
            snapshot.peak_pool_delta,
            snapshot.peak_kv_delta,
            snapshot.peak_kv_cache_delta,
        );

        for phase_snapshot in &snapshot.phases {
            if let Some(phase_stat) = self.phases.iter_mut().find(|phase| phase.label == phase_snapshot.label) {
                phase_stat.scope.update(
                    phase_snapshot.current_pool_delta,
                    phase_snapshot.current_kv_delta,
                    phase_snapshot.current_kv_cache_delta,
                    phase_snapshot.peak_pool_delta,
                    phase_snapshot.peak_kv_delta,
                    phase_snapshot.peak_kv_cache_delta,
                );
            } else {
                let mut scope = MemoryScopeStat::default();
                scope.update(
                    phase_snapshot.current_pool_delta,
                    phase_snapshot.current_kv_delta,
                    phase_snapshot.current_kv_cache_delta,
                    phase_snapshot.peak_pool_delta,
                    phase_snapshot.peak_kv_delta,
                    phase_snapshot.peak_kv_cache_delta,
                );
                self.phases.push(MemoryPhaseStat {
                    label: phase_snapshot.label.clone(),
                    scope,
                });
            }
        }

        self.phases
            .retain(|phase| snapshot.phases.iter().any(|snap| snap.label == phase.label));
    }
}

#[derive(Clone, Copy, Default)]
struct ScalarStat {
    baseline: Option<f64>,
    current: f64,
    peak: f64,
}

impl ScalarStat {
    fn record(&mut self, value: f64) {
        if self.baseline.is_none() {
            self.baseline = Some(value);
        }
        self.current = value;
        self.peak = self.peak.max(value);
    }

    fn baseline_mb(&self) -> Option<f64> {
        self.baseline
    }

    fn to_row(&self, label: &str) -> MemoryRow {
        MemoryRow {
            label: label.to_string(),
            level: 0,
            current_total_mb: self.current,
            peak_total_mb: self.peak,
            current_pool_mb: 0.0,
            peak_pool_mb: 0.0,
            current_kv_mb: 0.0,
            peak_kv_mb: 0.0,
            current_kv_cache_mb: 0.0,
            peak_kv_cache_mb: 0.0,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        }
    }
}

struct ProcessMemoryTracker {
    system: System,
    pid: Pid,
}

impl ProcessMemoryTracker {
    fn new() -> Option<Self> {
        let pid = get_current_pid().ok()?;
        let mut system = System::new();
        let targets = [pid];
        if system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&targets),
            false,
            ProcessRefreshKind::nothing().with_memory(),
        ) == 0
        {
            system.refresh_processes(ProcessesToUpdate::All, false);
        }
        Some(Self { system, pid })
    }

    fn sample_mb(&mut self) -> Option<f64> {
        let pid = self.pid;
        let targets = [pid];
        if self.system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&targets),
            false,
            ProcessRefreshKind::nothing().with_memory(),
        ) == 0
        {
            self.system.refresh_processes(ProcessesToUpdate::All, false);
        }
        self.system.process(pid).map(|process| process.memory() as f64 / 1024.0 / 1024.0)
    }
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn build_model_memory_tree(model: &Qwen25) -> ModelMemoryNode {
    let mut children = Vec::new();

    children.push(ModelMemoryNode::leaf("Token Embeddings", model.embed_weight.size_bytes()));
    children.push(ModelMemoryNode::leaf("Output Projection", model.output_weight.size_bytes()));
    children.push(ModelMemoryNode::leaf("Final Layer Norm", model.final_norm_gamma.size_bytes()));
    children.push(ModelMemoryNode::leaf(
        "RoPE Cache",
        model.rope_cos_cache.size_bytes() + model.rope_sin_cache.size_bytes(),
    ));

    let block_nodes: Vec<ModelMemoryNode> = model
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| {
            let attn_projections = ModelMemoryNode::branch(
                "Attention Projections",
                vec![
                    ModelMemoryNode::leaf("Q weight", block.attn_q_weight.size_bytes()),
                    ModelMemoryNode::leaf("K weight", block.attn_k_weight.size_bytes()),
                    ModelMemoryNode::leaf("V weight", block.attn_v_weight.size_bytes()),
                    ModelMemoryNode::leaf("Output weight", block.attn_out_weight.size_bytes()),
                ],
            );

            let attn_biases = ModelMemoryNode::branch(
                "Attention Biases",
                vec![
                    ModelMemoryNode::leaf("Q bias", block.attn_q_bias.size_bytes()),
                    ModelMemoryNode::leaf("K bias", block.attn_k_bias.size_bytes()),
                    ModelMemoryNode::leaf("V bias", block.attn_v_bias.size_bytes()),
                ],
            );

            let feedforward = ModelMemoryNode::branch(
                "Feedforward Projections",
                vec![
                    ModelMemoryNode::leaf("Gate weight", block.ffn_gate.size_bytes()),
                    ModelMemoryNode::leaf("Up weight", block.ffn_up.size_bytes()),
                    ModelMemoryNode::leaf("Down weight", block.ffn_down.size_bytes()),
                ],
            );

            let feedforward_biases = ModelMemoryNode::branch(
                "Feedforward Biases",
                vec![
                    ModelMemoryNode::leaf("Gate bias", block.ffn_gate_bias.size_bytes()),
                    ModelMemoryNode::leaf("Up bias", block.ffn_up_bias.size_bytes()),
                    ModelMemoryNode::leaf("Down bias", block.ffn_down_bias.size_bytes()),
                ],
            );

            let norms = ModelMemoryNode::branch(
                "Norm Parameters",
                vec![
                    ModelMemoryNode::leaf("Attention norm", block.attn_norm_gamma.size_bytes()),
                    ModelMemoryNode::leaf("FFN norm", block.ffn_norm_gamma.size_bytes()),
                ],
            );

            ModelMemoryNode::branch(
                format!("Block {}", idx + 1),
                vec![attn_projections, attn_biases, feedforward, feedforward_biases, norms],
            )
        })
        .collect();

    if !block_nodes.is_empty() {
        children.push(ModelMemoryNode::branch("Transformer Blocks", block_nodes));
    }

    ModelMemoryNode::branch("Model Weights", children)
}

fn append_model_memory_rows(node: &ModelMemoryNode, level: u8, rows: &mut Vec<MemoryRow>) {
    let total_mb = bytes_to_mb(node.total_bytes());
    if total_mb <= 0.0 {
        return;
    }

    rows.push(MemoryRow {
        label: node.label.clone(),
        level,
        current_total_mb: total_mb,
        peak_total_mb: total_mb,
        current_pool_mb: 0.0,
        peak_pool_mb: 0.0,
        current_kv_mb: 0.0,
        peak_kv_mb: 0.0,
        current_kv_cache_mb: 0.0,
        peak_kv_cache_mb: 0.0,
        absolute_pool_mb: 0.0,
        absolute_kv_mb: 0.0,
        absolute_kv_cache_mb: 0.0,
        show_absolute: false,
    });

    for child in &node.children {
        append_model_memory_rows(child, level + 1, rows);
    }
}

fn unattributed_host_row(host: &ScalarStat, tracked_mb: f64) -> Option<MemoryRow> {
    if host.current <= 0.0 {
        return None;
    }

    let current_gap = (host.current - tracked_mb).max(0.0);
    let peak_gap = (host.peak - tracked_mb).max(current_gap);
    if current_gap < 1.0 && peak_gap < 1.0 {
        return None;
    }

    Some(MemoryRow {
        label: "Unattributed Host Memory".to_string(),
        level: 1,
        current_total_mb: current_gap,
        peak_total_mb: peak_gap.max(current_gap),
        current_pool_mb: 0.0,
        peak_pool_mb: 0.0,
        current_kv_mb: 0.0,
        peak_kv_mb: 0.0,
        current_kv_cache_mb: 0.0,
        peak_kv_cache_mb: 0.0,
        absolute_pool_mb: 0.0,
        absolute_kv_mb: 0.0,
        absolute_kv_cache_mb: 0.0,
        show_absolute: false,
    })
}

fn baseline_host_row(host: &ScalarStat, tracked_mb: f64) -> Option<(MemoryRow, f64)> {
    let baseline = host.baseline_mb()?;
    if baseline <= 0.0 {
        return None;
    }

    let gap = (baseline - tracked_mb).max(0.0);
    if gap < 1.0 {
        return None;
    }

    let row = MemoryRow {
        label: "Process Baseline (Other)".to_string(),
        level: 1,
        current_total_mb: gap,
        peak_total_mb: gap,
        current_pool_mb: 0.0,
        peak_pool_mb: 0.0,
        current_kv_mb: 0.0,
        peak_kv_mb: 0.0,
        current_kv_cache_mb: 0.0,
        peak_kv_cache_mb: 0.0,
        absolute_pool_mb: 0.0,
        absolute_kv_mb: 0.0,
        absolute_kv_cache_mb: 0.0,
        show_absolute: false,
    };

    Some((row, gap))
}

fn append_reserved_pool_rows(usage: &MemoryUsage, rows: &mut Vec<MemoryRow>) {
    if usage.pool_capacity > 0 {
        let capacity_mb = bytes_to_mb(usage.pool_capacity);
        let used_mb = bytes_to_mb(usage.pool_used);
        rows.push(MemoryRow {
            label: "Tensor Pool (Reserved)".to_string(),
            level: 1,
            current_total_mb: capacity_mb,
            peak_total_mb: capacity_mb,
            current_pool_mb: used_mb,
            peak_pool_mb: used_mb,
            current_kv_mb: 0.0,
            peak_kv_mb: 0.0,
            current_kv_cache_mb: 0.0,
            peak_kv_cache_mb: 0.0,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        });
    }

    if usage.kv_capacity > 0 {
        let capacity_mb = bytes_to_mb(usage.kv_capacity);
        let used_mb = bytes_to_mb(usage.kv_used);
        let cache_mb = bytes_to_mb(usage.kv_cache_bytes);
        rows.push(MemoryRow {
            label: "KV Pool (Reserved)".to_string(),
            level: 1,
            current_total_mb: capacity_mb,
            peak_total_mb: capacity_mb,
            current_pool_mb: 0.0,
            peak_pool_mb: 0.0,
            current_kv_mb: used_mb,
            peak_kv_mb: used_mb,
            current_kv_cache_mb: cache_mb,
            peak_kv_cache_mb: cache_mb,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        });
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
    let output_tokens = generate_autoregressive_with_kv_cache(qwen, tokenizer, ctx, &input_ids, cfg, &[])?;
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
    host_overheads: &[(String, usize)],
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
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, &input_ids, cfg, &mut token_callback, tx, host_overheads)?;

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
    host_overheads: &[(String, usize)],
) -> Result<Vec<u32>, MetalError> {
    let mut result = Vec::new();
    let mut callback = |token_id, _decoded_token| -> Result<bool, MetalError> {
        result.push(token_id);
        Ok(true)
    };

    let (tx, _) = mpsc::channel();
    generate_autoregressive_with_kv_cache_streaming(qwen, tokenizer, ctx, input_ids, cfg, &mut callback, &tx, host_overheads)?;

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
    host_overheads: &[(String, usize)],
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
    let mut memory_embed = MemoryScopeStat::default();
    let mut memory_forward = MemoryScopeStat::default();
    let mut memory_output = MemoryScopeStat::default();
    let mut memory_blocks = vec![MemoryBlockStat::default(); n_layers];
    let mut latest_forward_usage: Option<MemoryUsage> = None;
    let mut memory_ready = false;
    let mut host_memory = ScalarStat::default();
    let mut process_memory_tracker = ProcessMemoryTracker::new();
    if let Some(tracker) = process_memory_tracker.as_mut() {
        if let Some(memory_mb) = tracker.sample_mb() {
            host_memory.record(memory_mb);
        }
    }
    let model_memory_tree = build_model_memory_tree(qwen);

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

        if let Some(tracker) = process_memory_tracker.as_mut() {
            if let Some(memory_mb) = tracker.sample_mb() {
                host_memory.record(memory_mb);
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

        if memory_ready && ui_connected {
            let rows = build_memory_rows(
                &model_memory_tree,
                &host_memory,
                &memory_embed,
                &memory_forward,
                latest_forward_usage,
                &memory_blocks,
                &memory_output,
                host_overheads,
            );
            if tx.send(AppEvent::MemoryUpdate(rows)).is_err() {
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

fn build_memory_rows(
    model_memory: &ModelMemoryNode,
    host: &ScalarStat,
    embed: &MemoryScopeStat,
    forward: &MemoryScopeStat,
    forward_usage: Option<MemoryUsage>,
    blocks: &[MemoryBlockStat],
    output: &MemoryScopeStat,
    host_overheads: &[(String, usize)],
) -> Vec<MemoryRow> {
    let mut rows = Vec::new();

    append_model_memory_rows(model_memory, 0, &mut rows);

    let mut tracked_mb = bytes_to_mb(model_memory.total_bytes());

    for (label, bytes) in host_overheads {
        if *bytes == 0 {
            continue;
        }
        let mb = bytes_to_mb(*bytes);
        tracked_mb += mb;
        rows.push(MemoryRow {
            label: label.clone(),
            level: 0,
            current_total_mb: mb,
            peak_total_mb: mb,
            current_pool_mb: 0.0,
            peak_pool_mb: 0.0,
            current_kv_mb: 0.0,
            peak_kv_mb: 0.0,
            current_kv_cache_mb: 0.0,
            peak_kv_cache_mb: 0.0,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        });
    }

    if host.current > 0.0 || host.peak > 0.0 {
        rows.push(host.to_row("Host Memory (MB)"));
        if let Some(usage) = forward_usage {
            tracked_mb += bytes_to_mb(usage.pool_capacity) + bytes_to_mb(usage.kv_capacity);
            append_reserved_pool_rows(&usage, &mut rows);
        }

        if let Some((baseline_row, baseline_mb)) = baseline_host_row(host, tracked_mb) {
            tracked_mb += baseline_mb;
            rows.push(baseline_row);
        }

        if let Some(unattributed) = unattributed_host_row(host, tracked_mb) {
            rows.push(unattributed);
        }
    }

    if embed.has_data() {
        rows.push(embed.to_row("Embedding".to_string(), 0));
    }

    if forward.has_data() || forward_usage.is_some() {
        let forward_row = forward.to_row("Forward Step".to_string(), 0);
        rows.push(forward_row);

        for (idx, block_stat) in blocks.iter().enumerate() {
            if !block_stat.scope.has_data() {
                continue;
            }
            rows.push(block_stat.scope.to_row(format!("Block {}", idx + 1), 1));
            for phase in &block_stat.phases {
                if !phase.scope.has_data() {
                    continue;
                }
                rows.push(phase.scope.to_row(phase.label.clone(), 2));
            }
        }
    }

    if output.has_data() {
        rows.push(output.to_row("Output".to_string(), 0));
    }

    rows
}
