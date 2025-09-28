use crate::app_event::{LatencyRow, MemoryRow};
use crate::metallic::instrumentation::{BlockMemorySnapshot, MemoryUsage};
use crate::metallic::kernels::softmax::SoftmaxBackend;
use crate::metallic::models::qwen25::Qwen25;
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use serde_json::json;
use std::{
    env,
    fs::OpenOptions,
    io::{self, BufWriter, Write},
    time::{Duration, Instant},
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};

pub const METRICS_LOG_INTERVAL_ENV: &str = "METRICS_LOG_INTERVAL_SECS";
pub const METRICS_LOG_ENABLED_ENV: &str = "METRICS_LOG_ENABLED";

#[derive(Clone)]
pub struct ModelMemoryNode {
    label: String,
    bytes: usize,
    children: Vec<ModelMemoryNode>,
}

impl ModelMemoryNode {
    pub fn leaf(label: impl Into<String>, bytes: usize) -> Self {
        Self {
            label: label.into(),
            bytes,
            children: Vec::new(),
        }
    }

    pub fn branch(label: impl Into<String>, children: Vec<ModelMemoryNode>) -> Self {
        Self {
            label: label.into(),
            bytes: 0,
            children,
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn children(&self) -> &[ModelMemoryNode] {
        &self.children
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes + self.children.iter().map(|child| child.total_bytes()).sum::<usize>()
    }
}

#[derive(Clone, Copy, Default)]
pub struct RollingStat {
    last: Duration,
    total: Duration,
    count: u64,
}

impl RollingStat {
    pub fn record(&mut self, duration: Duration) {
        self.last = duration;
        self.total += duration;
        self.count += 1;
    }

    pub fn last_ms(&self) -> f64 {
        self.last.as_secs_f64() * 1000.0
    }

    pub fn average_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.total.as_secs_f64() * 1000.0) / self.count as f64
        }
    }

    pub fn has_samples(&self) -> bool {
        self.count > 0
    }
}

#[derive(Clone, Default)]
pub struct SoftmaxBackendStats {
    kernel: RollingStat,
    mps: RollingStat,
}

impl SoftmaxBackendStats {
    pub fn record(&mut self, backend: SoftmaxBackend, duration: Duration) {
        match backend {
            SoftmaxBackend::Kernel => self.kernel.record(duration),
            SoftmaxBackend::Mps => self.mps.record(duration),
        }
    }

    pub fn kernel(&self) -> &RollingStat {
        &self.kernel
    }

    pub fn mps(&self) -> &RollingStat {
        &self.mps
    }
}

#[derive(Clone, Default)]
pub struct BlockPhaseStat {
    label: String,
    stat: RollingStat,
}

impl BlockPhaseStat {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn stat(&self) -> &RollingStat {
        &self.stat
    }
}

#[derive(Clone, Default)]
pub struct BlockStat {
    total: RollingStat,
    phases: Vec<BlockPhaseStat>,
}

impl BlockStat {
    pub fn record_total(&mut self, duration: Duration) {
        self.total.record(duration);
    }

    pub fn record_phase(&mut self, label: &str, duration: Duration) {
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

    pub fn total(&self) -> &RollingStat {
        &self.total
    }

    pub fn phases(&self) -> &[BlockPhaseStat] {
        &self.phases
    }
}

#[derive(Clone, Default)]
pub struct MemoryScopeStat {
    current_pool_mb: f64,
    peak_pool_mb: f64,
    current_kv_mb: f64,
    peak_kv_mb: f64,
    current_kv_cache_mb: f64,
    peak_kv_cache_mb: f64,
}

impl MemoryScopeStat {
    pub fn update(
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

    pub fn has_data(&self) -> bool {
        self.current_total_mb() > 0.0 || self.peak_total_mb() > 0.0
    }

    pub fn current_total_mb(&self) -> f64 {
        self.current_pool_mb + self.current_kv_mb + self.current_kv_cache_mb
    }

    pub fn peak_total_mb(&self) -> f64 {
        self.peak_pool_mb + self.peak_kv_mb + self.peak_kv_cache_mb
    }

    pub fn to_row(&self, label: String, level: u8) -> MemoryRow {
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
pub struct MemoryPhaseStat {
    label: String,
    scope: MemoryScopeStat,
}

impl MemoryPhaseStat {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn scope(&self) -> &MemoryScopeStat {
        &self.scope
    }
}

#[derive(Clone, Default)]
pub struct MemoryBlockStat {
    scope: MemoryScopeStat,
    phases: Vec<MemoryPhaseStat>,
}

impl MemoryBlockStat {
    pub fn update_from_snapshot(&mut self, snapshot: &BlockMemorySnapshot) {
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

    pub fn scope(&self) -> &MemoryScopeStat {
        &self.scope
    }

    pub fn phases(&self) -> &[MemoryPhaseStat] {
        &self.phases
    }
}

#[derive(Clone, Copy, Default)]
pub struct ScalarStat {
    baseline: Option<f64>,
    current: f64,
    peak: f64,
}

impl ScalarStat {
    pub fn record(&mut self, value: f64) {
        if self.baseline.is_none() {
            self.baseline = Some(value);
        }
        self.current = value;
        self.peak = self.peak.max(value);
    }

    pub fn baseline_mb(&self) -> Option<f64> {
        self.baseline
    }

    pub fn current(&self) -> f64 {
        self.current
    }

    pub fn peak(&self) -> f64 {
        self.peak
    }

    pub fn to_row(&self, label: &str) -> MemoryRow {
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

pub struct ProcessMemoryTracker {
    system: System,
    pid: Pid,
}

impl ProcessMemoryTracker {
    pub fn new() -> Option<Self> {
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

    pub fn sample_mb(&mut self) -> Option<f64> {
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

pub struct MetricsLoggers {
    memory: Option<JsonlLogger>,
    latency: Option<JsonlLogger>,
}

impl MetricsLoggers {
    pub fn from_env(interval: Duration) -> Option<Self> {
        if !logging_enabled() {
            return None;
        }

        let mut loggers = Self::new(interval).ok()?;
        if loggers.prime().is_err() {
            return None;
        }
        Some(loggers)
    }

    fn new(interval: Duration) -> io::Result<Self> {
        let timestamp = Utc::now().format("%Y%m%d%H%M%S").to_string();
        let memory = JsonlLogger::create(&format!("{}-memory.jsonl", timestamp), interval).ok();
        let latency = JsonlLogger::create(&format!("{}-latency.jsonl", timestamp), interval).ok();
        Ok(Self { memory, latency })
    }

    fn prime(&mut self) -> io::Result<()> {
        let now = Instant::now();
        if let Some(logger) = self.memory.as_mut() {
            logger.log::<MemoryRow>(&[], now, true)?;
            logger.last_logged = None;
        }
        if let Some(logger) = self.latency.as_mut() {
            logger.log::<LatencyRow>(&[], now, true)?;
            logger.last_logged = None;
        }
        Ok(())
    }

    pub fn log_memory(&mut self, rows: &[MemoryRow], now: Instant, force: bool) {
        if let Some(logger) = self.memory.as_mut()
            && let Err(err) = logger.log(rows, now, force)
        {
            eprintln!("Failed to log memory metrics: {err}");
            self.memory = None;
        }
    }

    pub fn log_latency(&mut self, rows: &[LatencyRow], now: Instant, force: bool) {
        if let Some(logger) = self.latency.as_mut()
            && let Err(err) = logger.log(rows, now, force)
        {
            eprintln!("Failed to log latency metrics: {err}");
            self.latency = None;
        }
    }
}

struct JsonlLogger {
    writer: BufWriter<std::fs::File>,
    interval: Duration,
    last_logged: Option<Instant>,
}

impl JsonlLogger {
    fn create(path: &str, interval: Duration) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            interval,
            last_logged: None,
        })
    }

    fn log<T>(&mut self, rows: &[T], now: Instant, force: bool) -> io::Result<()>
    where
        T: Serialize,
    {
        let should_log = force
            || self
                .last_logged
                .map(|last| now.duration_since(last) >= self.interval)
                .unwrap_or(true);

        if !should_log {
            return Ok(());
        }

        let entry = json!({
            "timestamp": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            "rows": rows,
        });
        serde_json::to_writer(&mut self.writer, &entry)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.last_logged = Some(now);
        Ok(())
    }
}

pub fn log_interval_from_env() -> Duration {
    env::var(METRICS_LOG_INTERVAL_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(5))
}

fn logging_enabled() -> bool {
    match env::var(METRICS_LOG_ENABLED_ENV) {
        Ok(value) => matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

pub fn build_latency_rows(
    embed: &RollingStat,
    forward: &RollingStat,
    blocks: &[BlockStat],
    softmax: &SoftmaxBackendStats,
    output: &RollingStat,
    sample: &RollingStat,
    decode: &RollingStat,
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

    if softmax.kernel().has_samples() {
        rows.push(LatencyRow {
            label: "Softmax (Kernel)".to_string(),
            last_ms: softmax.kernel().last_ms(),
            average_ms: softmax.kernel().average_ms(),
            level: 0,
        });
    }

    if softmax.mps().has_samples() {
        rows.push(LatencyRow {
            label: "Softmax (MPS)".to_string(),
            last_ms: softmax.mps().last_ms(),
            average_ms: softmax.mps().average_ms(),
            level: 0,
        });
    }

    for (idx, stat) in blocks.iter().enumerate() {
        rows.push(LatencyRow {
            label: format!("Block {}", idx + 1),
            last_ms: stat.total().last_ms(),
            average_ms: stat.total().average_ms(),
            level: 1,
        });
        for phase in stat.phases() {
            rows.push(LatencyRow {
                label: phase.label().to_string(),
                last_ms: phase.stat().last_ms(),
                average_ms: phase.stat().average_ms(),
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

    rows.push(LatencyRow {
        label: "Decode".to_string(),
        last_ms: decode.last_ms(),
        average_ms: decode.average_ms(),
        level: 0,
    });

    rows
}

#[allow(clippy::too_many_arguments)]
pub fn build_memory_rows(
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

    let mut host_tracked_mb = bytes_to_mb(model_memory.total_bytes());

    for (label, bytes) in host_overheads {
        if *bytes == 0 {
            continue;
        }
        let mb = bytes_to_mb(*bytes);
        host_tracked_mb += mb;
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

    if host.current() > 0.0 || host.peak() > 0.0 {
        let mut explained_host_mb = host_tracked_mb;
        rows.push(host.to_row("Host Memory (MB)"));
        if let Some(usage) = forward_usage {
            explained_host_mb += usage_tracked_mb(&usage);
            append_reserved_pool_rows(&usage, &mut rows);
        }

        if let Some((baseline_row, baseline_mb)) = baseline_host_row(host, explained_host_mb) {
            explained_host_mb += baseline_mb;
            rows.push(baseline_row);
        }

        if let Some(unattributed) = unattributed_host_row(host, explained_host_mb) {
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
            if !block_stat.scope().has_data() {
                continue;
            }
            rows.push(block_stat.scope().to_row(format!("Block {}", idx + 1), 1));
            for phase in block_stat.phases() {
                if !phase.scope().has_data() {
                    continue;
                }
                rows.push(phase.scope().to_row(phase.label().to_string(), 2));
            }
        }
    }

    if output.has_data() {
        rows.push(output.to_row("Output".to_string(), 0));
    }

    rows
}

pub fn build_model_memory_tree(model: &Qwen25) -> ModelMemoryNode {
    let mut children = vec![
        ModelMemoryNode::leaf("Token Embeddings", model.embed_weight.size_bytes()),
        ModelMemoryNode::leaf("Output Projection", model.output_weight.size_bytes()),
        ModelMemoryNode::leaf("Final Layer Norm", model.final_norm_gamma.size_bytes()),
        ModelMemoryNode::leaf("RoPE Cache", model.rope_cos_cache.size_bytes() + model.rope_sin_cache.size_bytes()),
    ];

    let block_nodes: Vec<ModelMemoryNode> = model
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| {
            let elem_bytes = std::mem::size_of::<f32>();
            let d_model = model.config.d_model;
            let kv_dim = block.kv_dim;
            let q_weight_bytes = d_model * d_model * elem_bytes;
            let k_weight_bytes = d_model * kv_dim * elem_bytes;
            let v_weight_bytes = d_model * kv_dim * elem_bytes;
            let q_bias_bytes = d_model * elem_bytes;
            let k_bias_bytes = kv_dim * elem_bytes;
            let v_bias_bytes = kv_dim * elem_bytes;

            let attn_projections = ModelMemoryNode::branch(
                "Attention Projections",
                vec![
                    ModelMemoryNode::leaf("Fused QKV weight", block.attn_qkv_weight.size_bytes()),
                    ModelMemoryNode::leaf("Q weight logical slice", q_weight_bytes),
                    ModelMemoryNode::leaf("K weight logical slice", k_weight_bytes),
                    ModelMemoryNode::leaf("V weight logical slice", v_weight_bytes),
                    ModelMemoryNode::leaf("Output weight", block.attn_out_weight.size_bytes()),
                ],
            );

            let attn_biases = ModelMemoryNode::branch(
                "Attention Biases",
                vec![
                    ModelMemoryNode::leaf("Fused QKV bias", block.attn_qkv_bias.size_bytes()),
                    ModelMemoryNode::leaf("Q bias logical slice", q_bias_bytes),
                    ModelMemoryNode::leaf("K bias logical slice", k_bias_bytes),
                    ModelMemoryNode::leaf("V bias logical slice", v_bias_bytes),
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
        label: node.label().to_string(),
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

    for child in node.children() {
        append_model_memory_rows(child, level + 1, rows);
    }
}

fn unattributed_host_row(host: &ScalarStat, tracked_mb: f64) -> Option<MemoryRow> {
    if host.current() <= 0.0 {
        return None;
    }

    let current_gap = (host.current() - tracked_mb).max(0.0);
    let peak_gap = (host.peak() - tracked_mb).max(current_gap);
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
    let explained = tracked_mb.min(baseline);
    if explained <= 0.0 {
        return None;
    }

    let row = MemoryRow {
        label: "Process Baseline (Other)".to_string(),
        level: 1,
        current_total_mb: explained,
        peak_total_mb: explained,
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

    Some((row, explained))
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
        rows.push(MemoryRow {
            label: "pool".to_string(),
            level: 2,
            current_total_mb: used_mb,
            peak_total_mb: used_mb,
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
            current_kv_mb: 0.0,
            peak_kv_mb: 0.0,
            current_kv_cache_mb: 0.0,
            peak_kv_cache_mb: 0.0,
            absolute_pool_mb: 0.0,
            absolute_kv_mb: 0.0,
            absolute_kv_cache_mb: 0.0,
            show_absolute: false,
        });
        rows.push(MemoryRow {
            label: "kv".to_string(),
            level: 2,
            current_total_mb: used_mb,
            peak_total_mb: used_mb,
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
        rows.push(MemoryRow {
            label: "kv-cache".to_string(),
            level: 2,
            current_total_mb: cache_mb,
            peak_total_mb: cache_mb,
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
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn usage_tracked_mb(usage: &MemoryUsage) -> f64 {
    let pool_mb = bytes_to_mb(usage.pool_used.max(usage.pool_capacity));
    let kv_mb = bytes_to_mb(usage.kv_used.max(usage.kv_capacity));
    let kv_cache_mb = bytes_to_mb(usage.kv_cache_bytes);
    pool_mb + kv_mb + kv_cache_mb
}
