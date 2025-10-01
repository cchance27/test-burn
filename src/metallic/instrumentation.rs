use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use rustc_hash::FxHashMap;

use crate::metallic::kernels::matmul::MatMulBackend;
#[cfg(test)]
use crate::metallic::kernels::matmul::MatMulSample;
use crate::metallic::operation::CommandBuffer;

/// Handle to a shared latency collector used to instrument fine-grained timing inside
/// the Metal execution context. The collector is populated by `Context` while the
/// inference loops execute and later inspected by higher-level orchestration code.

pub type LatencyCollectorHandle = Rc<RefCell<StepLatencyCollector>>;
pub type MemoryCollectorHandle = Rc<RefCell<StepMemoryCollector>>;

/// Shared recorder handle that allows matmul dispatches to append timing samples
/// once the GPU finishes executing a command buffer.
#[derive(Clone)]
pub struct MatMulSampleRecorder {
    inner: Arc<dyn Fn(MatMulBackend, Duration) + Send + Sync>,
}

impl MatMulSampleRecorder {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(MatMulBackend, Duration) + Send + Sync + 'static,
    {
        Self { inner: Arc::new(callback) }
    }

    pub fn record_matmul_backend_sample(&self, backend: MatMulBackend, duration: Duration) {
        (self.inner)(backend, duration);
    }
}

/// Tracks in-flight matmul dispatches so their GPU execution time can be
/// recorded once the surrounding command buffer completes.
#[derive(Clone, Default)]
pub struct MatMulInstrumentation {
    inner: Arc<MatMulInstrumentationInner>,
}

struct MatMulInstrumentationInner {
    pending: Mutex<FxHashMap<usize, PendingMatMul>>,
}

impl Default for MatMulInstrumentationInner {
    fn default() -> Self {
        Self {
            pending: Mutex::new(FxHashMap::default()),
        }
    }
}

struct PendingMatMul {
    recorder: MatMulSampleRecorder,
    counts: FxHashMap<MatMulBackend, usize>,
}

impl PendingMatMul {
    fn new(recorder: MatMulSampleRecorder) -> Self {
        Self {
            recorder,
            counts: FxHashMap::default(),
        }
    }

    fn increment(&mut self, backend: MatMulBackend) {
        *self.counts.entry(backend).or_insert(0) += 1;
    }
}

impl MatMulInstrumentation {
    fn lock_pending(&self) -> MutexGuard<'_, FxHashMap<usize, PendingMatMul>> {
        self.inner.pending.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn register(&self, command_buffer: &CommandBuffer, backend: MatMulBackend, recorder: MatMulSampleRecorder) {
        {
            let mut pending = self.lock_pending();
            if let Some(entry) = pending.get_mut(&Self::buffer_key(command_buffer)) {
                entry.increment(backend);
                return;
            }
        }

        let key = Self::buffer_key(command_buffer);
        self.install_completion(command_buffer, key);

        let mut entry = PendingMatMul::new(recorder);
        entry.increment(backend);
        self.lock_pending().insert(key, entry);
    }

    fn install_completion(&self, command_buffer: &CommandBuffer, key: usize) {
        let instrumentation = self.clone();
        command_buffer.on_completed(move |raw| instrumentation.complete(key, raw));
    }

    fn complete(&self, key: usize, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        let Some(entry) = self.lock_pending().remove(&key) else {
            return;
        };

        // GPU timing information is reported in seconds.
        let gpu_start = unsafe { command_buffer.GPUStartTime() };
        let gpu_end = unsafe { command_buffer.GPUEndTime() };

        if !gpu_start.is_finite() || !gpu_end.is_finite() {
            return;
        }

        let delta = gpu_end - gpu_start;
        if delta <= 0.0 {
            return;
        }

        let total = Duration::from_secs_f64(delta);
        Self::dispatch_samples(entry, total);
    }

    fn dispatch_samples(entry: PendingMatMul, total: Duration) {
        let PendingMatMul { recorder, counts } = entry;
        let total_secs = total.as_secs_f64();
        if total_secs <= 0.0 {
            return;
        }

        for (backend, count) in counts {
            if count == 0 {
                continue;
            }

            let slice_secs = total_secs / count as f64;
            if !slice_secs.is_finite() || slice_secs <= 0.0 {
                continue;
            }

            recorder.record_matmul_backend_sample(backend, Duration::from_secs_f64(slice_secs));
        }
    }

    fn buffer_key(command_buffer: &CommandBuffer) -> usize {
        Retained::as_ptr(command_buffer.raw()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn dispatch_samples_splits_duration_evenly() {
        let samples: Arc<Mutex<Vec<MatMulSample>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = MatMulSampleRecorder::new({
            let samples = Arc::clone(&samples);
            move |backend, duration| {
                if let Ok(mut samples) = samples.lock() {
                    samples.push(MatMulSample { backend, duration });
                }
            }
        });

        let mut pending = PendingMatMul::new(recorder);
        pending.increment(MatMulBackend::Mps);
        pending.increment(MatMulBackend::Mps);

        MatMulInstrumentation::dispatch_samples(pending, Duration::from_millis(30));

        let recorded = samples.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        for sample in recorded.iter() {
            assert_eq!(sample.backend, MatMulBackend::Mps);
            assert!((sample.duration.as_secs_f64() - 0.015).abs() < 1e-6);
        }
    }
}

/// Enumeration of the latency events that can be emitted from the low-level kernels.
#[derive(Clone, Debug)]
pub enum LatencyEvent<'a> {
    ForwardStep,
    Block { index: usize },
    BlockPhase { index: usize, label: Cow<'a, str> },
}

impl<'a> LatencyEvent<'a> {
    pub fn block_phase<T>(index: usize, label: T) -> Self
    where
        T: Into<Cow<'a, str>>,
    {
        LatencyEvent::BlockPhase {
            index,
            label: label.into(),
        }
    }
}

/// Per-token latency collector that stores the most recent measurements for the
/// surrounding forward step as well as each transformer block.
#[derive(Debug)]
pub struct StepLatencyCollector {
    forward_step: Option<Duration>,
    block_durations: Vec<BlockLatencyEntry>,
}

impl StepLatencyCollector {
    /// Create a collector that can store measurements for `block_count` transformer blocks.
    pub fn new(block_count: usize) -> Self {
        Self {
            forward_step: None,
            block_durations: vec![BlockLatencyEntry::default(); block_count],
        }
    }

    /// Update the collector with a new latency measurement for the given event.
    pub fn record(&mut self, event: LatencyEvent<'_>, duration: Duration) {
        match event {
            LatencyEvent::ForwardStep => {
                self.forward_step = Some(duration);
            }
            LatencyEvent::Block { index } => {
                if let Some(slot) = self.block_durations.get_mut(index) {
                    slot.total = Some(duration);
                }
            }
            LatencyEvent::BlockPhase { index, label } => {
                if let Some(slot) = self.block_durations.get_mut(index) {
                    let label_owned = label.into_owned();
                    if let Some(existing) = slot.phases.iter_mut().find(|phase| phase.label == label_owned) {
                        existing.duration = duration;
                    } else {
                        slot.phases.push(BlockPhaseEntry {
                            label: label_owned,
                            duration,
                        });
                    }
                }
            }
        }
    }

    /// Read the most recent measurements without consuming the collector.
    pub fn snapshot(&self) -> StepLatencySnapshot {
        StepLatencySnapshot {
            forward_step: self.forward_step.unwrap_or_default(),
            blocks: self.block_durations.iter().map(BlockLatencyEntry::snapshot).collect(),
        }
    }
}

/// Copy of the most recent timings recorded for a single iteration.
#[derive(Clone, Debug, Default)]
pub struct StepLatencySnapshot {
    pub forward_step: Duration,
    pub blocks: Vec<BlockLatencySnapshot>,
}

impl StepLatencySnapshot {
    /// Convenience to create an empty snapshot with space for `block_count` blocks.
    pub fn empty(block_count: usize) -> Self {
        Self {
            forward_step: Duration::default(),
            blocks: vec![BlockLatencySnapshot::default(); block_count],
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlockLatencyEntry {
    total: Option<Duration>,
    phases: Vec<BlockPhaseEntry>,
}

impl BlockLatencyEntry {
    fn snapshot(&self) -> BlockLatencySnapshot {
        BlockLatencySnapshot {
            total: self.total.unwrap_or_default(),
            phases: self
                .phases
                .iter()
                .map(|phase| BlockPhaseSnapshot {
                    label: phase.label.clone(),
                    duration: phase.duration,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct BlockPhaseEntry {
    label: String,
    duration: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct BlockLatencySnapshot {
    pub total: Duration,
    pub phases: Vec<BlockPhaseSnapshot>,
}

#[derive(Clone, Debug)]
pub struct BlockPhaseSnapshot {
    pub label: String,
    pub duration: Duration,
}

/// Helper to create a new collector handle for the desired number of transformer blocks.
pub fn new_latency_collector(block_count: usize) -> LatencyCollectorHandle {
    Rc::new(RefCell::new(StepLatencyCollector::new(block_count)))
}

// --- Memory Instrumentation ---

#[derive(Clone, Debug)]
pub enum MemoryEvent<'a> {
    ForwardStart,
    ForwardSample,
    BlockStart { index: usize },
    BlockEnd { index: usize },
    BlockPhase { index: usize, label: Cow<'a, str> },
}

impl<'a> MemoryEvent<'a> {
    pub fn block_phase<T>(index: usize, label: T) -> Self
    where
        T: Into<Cow<'a, str>>,
    {
        MemoryEvent::BlockPhase {
            index,
            label: label.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryUsage {
    pub pool_used: usize,
    pub pool_capacity: usize,
    pub kv_used: usize,
    pub kv_capacity: usize,
    pub kv_cache_bytes: usize,
}

impl MemoryUsage {
    pub fn delta_from(self, baseline: MemoryUsage) -> MemoryUsageDelta {
        MemoryUsageDelta {
            pool_used: self.pool_used.saturating_sub(baseline.pool_used),
            kv_used: self.kv_used.saturating_sub(baseline.kv_used),
            kv_cache_bytes: self.kv_cache_bytes.saturating_sub(baseline.kv_cache_bytes),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryUsageDelta {
    pub pool_used: usize,
    pub kv_used: usize,
    pub kv_cache_bytes: usize,
}

impl MemoryUsageDelta {
    pub fn total_bytes(&self) -> usize {
        self.pool_used + self.kv_used
    }
}

#[derive(Debug)]
pub struct StepMemoryCollector {
    forward: MemoryScopeEntry,
    blocks: Vec<BlockMemoryEntry>,
}

impl StepMemoryCollector {
    pub fn new(block_count: usize) -> Self {
        Self {
            forward: MemoryScopeEntry::default(),
            blocks: vec![BlockMemoryEntry::default(); block_count],
        }
    }

    pub fn record(&mut self, event: MemoryEvent<'_>, usage: MemoryUsage) {
        match event {
            MemoryEvent::ForwardStart => {
                self.forward.record_baseline(usage);
                for block in &mut self.blocks {
                    block.reset();
                }
            }
            MemoryEvent::ForwardSample => {
                self.forward.record_sample(usage);
            }
            MemoryEvent::BlockStart { index } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_baseline(usage);
                }
                self.forward.record_sample(usage);
            }
            MemoryEvent::BlockEnd { index } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_sample(usage);
                }
                self.forward.record_sample(usage);
            }
            MemoryEvent::BlockPhase { index, label } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_phase(label.into_owned(), usage);
                }
                self.forward.record_sample(usage);
            }
        }
    }

    pub fn snapshot(&self) -> StepMemorySnapshot {
        StepMemorySnapshot {
            forward: self.forward.snapshot(),
            blocks: self.blocks.iter().map(BlockMemoryEntry::snapshot).collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct MemoryScopeEntry {
    baseline: Option<MemoryUsage>,
    last: Option<MemoryUsage>,
    peak_pool_delta: usize,
    peak_kv_delta: usize,
    peak_kv_cache_delta: usize,
}

impl MemoryScopeEntry {
    fn record_baseline(&mut self, usage: MemoryUsage) {
        self.baseline = Some(usage);
        self.last = Some(usage);
        self.peak_pool_delta = 0;
        self.peak_kv_delta = 0;
        self.peak_kv_cache_delta = 0;
    }

    fn record_sample(&mut self, usage: MemoryUsage) {
        if self.baseline.is_none() {
            self.record_baseline(usage);
            return;
        }
        self.last = Some(usage);
        let delta = usage.delta_from(self.baseline.unwrap());
        self.peak_pool_delta = self.peak_pool_delta.max(delta.pool_used);
        self.peak_kv_delta = self.peak_kv_delta.max(delta.kv_used);
        self.peak_kv_cache_delta = self.peak_kv_cache_delta.max(delta.kv_cache_bytes);
    }

    fn snapshot(&self) -> MemoryScopeSnapshot {
        let (current_pool_delta, current_kv_delta, current_kv_cache_delta) = if let (Some(base), Some(last)) = (self.baseline, self.last) {
            let delta = last.delta_from(base);
            (delta.pool_used, delta.kv_used, delta.kv_cache_bytes)
        } else {
            (0, 0, 0)
        };

        MemoryScopeSnapshot {
            baseline: self.baseline,
            last: self.last,
            current_pool_delta,
            current_kv_delta,
            current_kv_cache_delta,
            peak_pool_delta: self.peak_pool_delta,
            peak_kv_delta: self.peak_kv_delta,
            peak_kv_cache_delta: self.peak_kv_cache_delta,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlockMemoryEntry {
    scope: MemoryScopeEntry,
    phases: Vec<MemoryPhaseEntry>,
}

impl BlockMemoryEntry {
    fn reset(&mut self) {
        self.scope.baseline = None;
        self.scope.last = None;
        self.scope.peak_kv_delta = 0;
        self.scope.peak_pool_delta = 0;
        self.scope.peak_kv_cache_delta = 0;
        self.phases.clear();
    }

    fn record_baseline(&mut self, usage: MemoryUsage) {
        self.scope.record_baseline(usage);
        self.phases.clear();
    }

    fn record_sample(&mut self, usage: MemoryUsage) {
        self.scope.record_sample(usage);
    }

    fn record_phase(&mut self, label: String, usage: MemoryUsage) {
        self.scope.record_sample(usage);
        let baseline = self.scope.baseline.unwrap_or(usage);
        let delta = usage.delta_from(baseline);
        if let Some(phase) = self.phases.iter_mut().find(|phase| phase.label == label) {
            phase.usage = usage;
            phase.current_pool_delta = delta.pool_used;
            phase.current_kv_delta = delta.kv_used;
            phase.current_kv_cache_delta = delta.kv_cache_bytes;
            phase.peak_pool_delta = phase.peak_pool_delta.max(delta.pool_used);
            phase.peak_kv_delta = phase.peak_kv_delta.max(delta.kv_used);
            phase.peak_kv_cache_delta = phase.peak_kv_cache_delta.max(delta.kv_cache_bytes);
        } else {
            self.phases.push(MemoryPhaseEntry {
                label,
                usage,
                current_pool_delta: delta.pool_used,
                current_kv_delta: delta.kv_used,
                current_kv_cache_delta: delta.kv_cache_bytes,
                peak_pool_delta: delta.pool_used,
                peak_kv_delta: delta.kv_used,
                peak_kv_cache_delta: delta.kv_cache_bytes,
            });
        }
    }

    fn snapshot(&self) -> BlockMemorySnapshot {
        let scope = self.scope.snapshot();
        BlockMemorySnapshot {
            baseline: scope.baseline,
            last: scope.last,
            current_pool_delta: scope.current_pool_delta,
            current_kv_delta: scope.current_kv_delta,
            current_kv_cache_delta: scope.current_kv_cache_delta,
            peak_pool_delta: scope.peak_pool_delta,
            peak_kv_delta: scope.peak_kv_delta,
            peak_kv_cache_delta: scope.peak_kv_cache_delta,
            phases: self.phases.iter().map(MemoryPhaseEntry::snapshot).collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct MemoryPhaseEntry {
    label: String,
    usage: MemoryUsage,
    current_pool_delta: usize,
    current_kv_delta: usize,
    current_kv_cache_delta: usize,
    peak_pool_delta: usize,
    peak_kv_delta: usize,
    peak_kv_cache_delta: usize,
}

impl MemoryPhaseEntry {
    fn snapshot(&self) -> MemoryPhaseSnapshot {
        MemoryPhaseSnapshot {
            label: self.label.clone(),
            current_pool_delta: self.current_pool_delta,
            current_kv_delta: self.current_kv_delta,
            current_kv_cache_delta: self.current_kv_cache_delta,
            peak_pool_delta: self.peak_pool_delta,
            peak_kv_delta: self.peak_kv_delta,
            peak_kv_cache_delta: self.peak_kv_cache_delta,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryScopeSnapshot {
    pub baseline: Option<MemoryUsage>,
    pub last: Option<MemoryUsage>,
    pub current_pool_delta: usize,
    pub current_kv_delta: usize,
    pub current_kv_cache_delta: usize,
    pub peak_pool_delta: usize,
    pub peak_kv_delta: usize,
    pub peak_kv_cache_delta: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BlockMemorySnapshot {
    pub baseline: Option<MemoryUsage>,
    pub last: Option<MemoryUsage>,
    pub current_pool_delta: usize,
    pub current_kv_delta: usize,
    pub current_kv_cache_delta: usize,
    pub peak_pool_delta: usize,
    pub peak_kv_delta: usize,
    pub peak_kv_cache_delta: usize,
    pub phases: Vec<MemoryPhaseSnapshot>,
}

#[derive(Clone, Debug)]
pub struct MemoryPhaseSnapshot {
    pub label: String,
    pub current_pool_delta: usize,
    pub current_kv_delta: usize,
    pub current_kv_cache_delta: usize,
    pub peak_pool_delta: usize,
    pub peak_kv_delta: usize,
    pub peak_kv_cache_delta: usize,
}

#[derive(Clone, Debug, Default)]
pub struct StepMemorySnapshot {
    pub forward: MemoryScopeSnapshot,
    pub blocks: Vec<BlockMemorySnapshot>,
}

impl StepMemorySnapshot {
    pub fn empty(block_count: usize) -> Self {
        Self {
            forward: MemoryScopeSnapshot::default(),
            blocks: vec![BlockMemorySnapshot::default(); block_count],
        }
    }
}

pub fn new_memory_collector(block_count: usize) -> MemoryCollectorHandle {
    Rc::new(RefCell::new(StepMemoryCollector::new(block_count)))
}
