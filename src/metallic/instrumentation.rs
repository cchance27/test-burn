use std::borrow::Cow;
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
#[cfg(target_os = "macos")]
use std::thread;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSUInteger};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
    MTLStorageMode,
};
use rustc_hash::FxHashMap;

use crate::metallic::kernels::matmul::MatMulBackend;
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

const COUNTER_SAMPLE_CAPACITY: usize = 512;

/// Tracks in-flight matmul dispatches so their GPU execution time can be
/// recorded once the surrounding command buffer completes.
#[derive(Clone)]
pub struct MatMulInstrumentation {
    inner: Arc<MatMulInstrumentationInner>,
}

#[derive(Clone, Copy)]
struct CounterSamplingSupport {
    compute_dispatch: bool,
    blit_boundary: bool,
}

impl CounterSamplingSupport {
    const fn disabled() -> Self {
        Self {
            compute_dispatch: false,
            blit_boundary: false,
        }
    }

    fn supports_backend(self, backend: MatMulBackend) -> bool {
        match backend {
            MatMulBackend::Mps => self.blit_boundary,
            MatMulBackend::Mlx | MatMulBackend::MlxTransposed => self.compute_dispatch,
            MatMulBackend::Total => false,
        }
    }

    const fn requires_counter_buffer(self) -> bool {
        self.compute_dispatch || self.blit_boundary
    }
}

#[cfg(target_os = "macos")]
fn query_counter_sampling_support(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> CounterSamplingSupport {
    CounterSamplingSupport {
        compute_dispatch: device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
        blit_boundary: device.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary),
    }
}

#[cfg(not(target_os = "macos"))]
fn query_counter_sampling_support(_device: &Retained<ProtocolObject<dyn MTLDevice>>) -> CounterSamplingSupport {
    CounterSamplingSupport::disabled()
}

struct MatMulInstrumentationInner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    counter_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
    gpu_timestamp_period: Option<f64>,
    sampling_support: CounterSamplingSupport,
    pending: Mutex<FxHashMap<usize, CommandBufferInstrumentation>>,
}

impl MatMulInstrumentationInner {
    fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let counter_set = Self::resolve_timestamp_counter_set(device);
        let gpu_timestamp_period = compute_gpu_timestamp_period(device);
        let sampling_support = query_counter_sampling_support(device);

        Self {
            device: device.clone(),
            counter_set,
            gpu_timestamp_period,
            sampling_support,
            pending: Mutex::new(FxHashMap::default()),
        }
    }

    fn resolve_timestamp_counter_set(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
        unsafe {
            let counter_sets = device.counterSets()?;
            let desired = (&*MTLCommonCounterSetTimestamp).to_string();
            let count = counter_sets.count() as usize;
            for idx in 0..count {
                let set = counter_sets.objectAtIndex(idx as NSUInteger);
                if set.name().to_string() == desired {
                    return Some(set);
                }
            }
            None
        }
    }

    fn create_sample_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
        if !self.sampling_support.requires_counter_buffer() {
            return None;
        }

        let counter_set = self.counter_set.as_ref()?;
        let descriptor = unsafe { MTLCounterSampleBufferDescriptor::new() };
        unsafe {
            descriptor.setCounterSet(Some(counter_set));
            descriptor.setSampleCount(COUNTER_SAMPLE_CAPACITY as NSUInteger);
            descriptor.setStorageMode(MTLStorageMode::Shared);
            self.device.newCounterSampleBufferWithDescriptor_error(&descriptor).ok()
        }
    }

    fn create_entry(&self, recorder: MatMulSampleRecorder) -> CommandBufferInstrumentation {
        let sample_buffer = self.create_sample_buffer();
        CommandBufferInstrumentation::new(recorder, sample_buffer, self.gpu_timestamp_period, self.sampling_support)
    }

    fn sample_buffer_for_key(&self, key: usize) -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
        let guard = self.pending.lock().ok()?;
        guard.get(&key).and_then(|entry| entry.sample_buffer.as_ref().map(Retained::clone))
    }
}

impl MatMulInstrumentation {
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            inner: Arc::new(MatMulInstrumentationInner::new(device)),
        }
    }

    fn lock_pending(&self) -> MutexGuard<'_, FxHashMap<usize, CommandBufferInstrumentation>> {
        self.inner.pending.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn register(&self, command_buffer: &CommandBuffer, backend: MatMulBackend, recorder: MatMulSampleRecorder) -> MatMulDispatchHandle {
        let key = Self::buffer_key(command_buffer);
        if let Some(allocation) = {
            let mut pending = self.lock_pending();
            pending.get_mut(&key).map(|entry| entry.register_dispatch(backend))
        } {
            return MatMulDispatchHandle::new(self.clone(), key, allocation);
        }

        self.install_completion(command_buffer, key);

        let mut entry = self.inner.create_entry(recorder);
        let allocation = entry.register_dispatch(backend);
        self.lock_pending().insert(key, entry);

        MatMulDispatchHandle::new(self.clone(), key, allocation)
    }

    fn install_completion(&self, command_buffer: &CommandBuffer, key: usize) {
        let instrumentation = self.clone();
        command_buffer.on_completed(move |raw| instrumentation.complete(key, raw));
    }

    fn complete(&self, key: usize, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        let entry = self.lock_pending().remove(&key);
        if let Some(entry) = entry {
            entry.finalize(command_buffer);
        }
    }

    fn sample_compute(&self, key: usize, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize) {
        if !self.inner.sampling_support.compute_dispatch {
            return;
        }

        if let Some(sample_buffer) = self.inner.sample_buffer_for_key(key) {
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(&sample_buffer, index as NSUInteger, true);
            }
        }
    }

    fn sample_blit(&self, key: usize, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>, index: usize) {
        if !self.inner.sampling_support.blit_boundary {
            return;
        }

        let Some(sample_buffer) = self.inner.sample_buffer_for_key(key) else {
            return;
        };
        let Some(encoder) = command_buffer.blitCommandEncoder() else {
            return;
        };
        unsafe {
            encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(&sample_buffer, index as NSUInteger, true);
        }
        encoder.endEncoding();
    }

    fn buffer_key(command_buffer: &CommandBuffer) -> usize {
        Retained::as_ptr(command_buffer.raw()) as usize
    }
}

#[derive(Clone)]
pub struct MatMulDispatchHandle {
    instrumentation: MatMulInstrumentation,
    key: usize,
    start_index: Option<usize>,
    end_index: Option<usize>,
}

impl MatMulDispatchHandle {
    fn new(instrumentation: MatMulInstrumentation, key: usize, allocation: DispatchAllocation) -> Self {
        Self {
            instrumentation,
            key,
            start_index: allocation.start_index,
            end_index: allocation.end_index,
        }
    }

    pub fn sample_start_compute(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        if let Some(index) = self.start_index {
            self.instrumentation.sample_compute(self.key, encoder, index);
        }
    }

    pub fn sample_end_compute(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        if let Some(index) = self.end_index {
            self.instrumentation.sample_compute(self.key, encoder, index);
        }
    }

    pub fn sample_start_blit(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        if let Some(index) = self.start_index {
            self.instrumentation.sample_blit(self.key, command_buffer, index);
        }
    }

    pub fn sample_end_blit(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        if let Some(index) = self.end_index {
            self.instrumentation.sample_blit(self.key, command_buffer, index);
        }
    }
}

struct DispatchAllocation {
    start_index: Option<usize>,
    end_index: Option<usize>,
}

struct DispatchTiming {
    backend: MatMulBackend,
    start_index: Option<usize>,
    end_index: Option<usize>,
}

struct CommandBufferInstrumentation {
    recorder: MatMulSampleRecorder,
    sample_buffer: Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
    dispatches: Vec<DispatchTiming>,
    used_samples: usize,
    gpu_timestamp_period: Option<f64>,
    sampling_support: CounterSamplingSupport,
}

impl CommandBufferInstrumentation {
    fn new(
        recorder: MatMulSampleRecorder,
        sample_buffer: Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
        gpu_timestamp_period: Option<f64>,
        sampling_support: CounterSamplingSupport,
    ) -> Self {
        Self {
            recorder,
            sample_buffer,
            dispatches: Vec::new(),
            used_samples: 0,
            gpu_timestamp_period,
            sampling_support,
        }
    }

    fn register_dispatch(&mut self, backend: MatMulBackend) -> DispatchAllocation {
        let (start_index, end_index) = if self.sampling_support.supports_backend(backend) {
            if let Some(buffer) = &self.sample_buffer {
                let capacity = unsafe { buffer.sampleCount() as usize };
                if self.used_samples + 1 < capacity {
                    let start = self.used_samples;
                    let end = self.used_samples + 1;
                    self.used_samples += 2;
                    (Some(start), Some(end))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        self.dispatches.push(DispatchTiming {
            backend,
            start_index,
            end_index,
        });

        DispatchAllocation { start_index, end_index }
    }

    fn finalize(self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
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
        let CommandBufferInstrumentation {
            recorder,
            sample_buffer,
            dispatches,
            used_samples,
            gpu_timestamp_period,
            ..
        } = self;

        recorder.record_matmul_backend_sample(MatMulBackend::Total, total);

        if dispatches.is_empty() {
            return;
        }

        if let (Some(sample_buffer), Some(period)) = (sample_buffer, gpu_timestamp_period) {
            if used_samples == 0 {
                return;
            }

            let Some(resolved) = (unsafe { sample_buffer.resolveCounterRange(NSRange::new(0, used_samples)) }) else {
                return;
            };

            let bytes = unsafe { resolved.as_bytes_unchecked() };
            let expected = used_samples * std::mem::size_of::<MTLCounterResultTimestamp>();
            if bytes.len() < expected {
                return;
            }

            let timestamps = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const MTLCounterResultTimestamp, used_samples) };

            let mut totals: FxHashMap<MatMulBackend, Duration> = FxHashMap::default();

            for dispatch in dispatches {
                let (Some(start), Some(end)) = (dispatch.start_index, dispatch.end_index) else {
                    continue;
                };
                if end >= timestamps.len() || start >= timestamps.len() {
                    continue;
                }
                let start_tick = timestamps[start].timestamp;
                let end_tick = timestamps[end].timestamp;
                if end_tick <= start_tick {
                    continue;
                }

                let duration_secs = (end_tick - start_tick) as f64 * period;
                if !duration_secs.is_finite() || duration_secs <= 0.0 {
                    continue;
                }

                let duration = Duration::from_secs_f64(duration_secs);
                totals.entry(dispatch.backend).and_modify(|d| *d += duration).or_insert(duration);
            }

            for (backend, duration) in totals {
                recorder.record_matmul_backend_sample(backend, duration);
            }

            return;
        }

        Self::record_fallback_backend_totals(&recorder, &dispatches, total);
    }
}

impl CommandBufferInstrumentation {
    fn record_fallback_backend_totals(recorder: &MatMulSampleRecorder, dispatches: &[DispatchTiming], total: Duration) {
        let mut backend_counts: FxHashMap<MatMulBackend, usize> = FxHashMap::default();
        for dispatch in dispatches {
            *backend_counts.entry(dispatch.backend).or_insert(0) += 1;
        }

        let total_dispatches: usize = backend_counts.values().sum();
        if total_dispatches == 0 {
            return;
        }

        let total_secs = total.as_secs_f64();
        if !total_secs.is_finite() || total_secs <= 0.0 {
            return;
        }

        let per_dispatch = total_secs / total_dispatches as f64;
        if !per_dispatch.is_finite() || per_dispatch <= 0.0 {
            return;
        }

        for (backend, count) in backend_counts {
            let duration_secs = per_dispatch * count as f64;
            if duration_secs <= 0.0 || !duration_secs.is_finite() {
                continue;
            }

            let duration = Duration::from_secs_f64(duration_secs);
            if duration.is_zero() {
                continue;
            }

            recorder.record_matmul_backend_sample(backend, duration);
        }
    }
}

#[cfg(target_os = "macos")]
fn compute_gpu_timestamp_period(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Option<f64> {
    unsafe {
        let mut cpu0: u64 = 0;
        let mut gpu0: u64 = 0;
        device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu0), NonNull::from(&mut gpu0));

        thread::sleep(Duration::from_millis(1));

        let mut cpu1: u64 = 0;
        let mut gpu1: u64 = 0;
        device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu1), NonNull::from(&mut gpu1));

        let cpu_delta = cpu1.checked_sub(cpu0)?;
        let gpu_delta = gpu1.checked_sub(gpu0)?;
        if cpu_delta == 0 || gpu_delta == 0 {
            return None;
        }

        let cpu_seconds = mach_time_to_seconds(cpu_delta)?;
        Some(cpu_seconds / gpu_delta as f64)
    }
}

#[cfg(not(target_os = "macos"))]
fn compute_gpu_timestamp_period(_device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Option<f64> {
    None
}

#[cfg(target_os = "macos")]
fn mach_time_to_seconds(delta: u64) -> Option<f64> {
    let (numer, denom) = mach_timebase_ratio()?;
    let nanos = (delta as u128).checked_mul(numer as u128)?.checked_div(denom as u128)?;
    Some(nanos as f64 / 1_000_000_000.0)
}

#[cfg(target_os = "macos")]
fn mach_timebase_ratio() -> Option<(u32, u32)> {
    static INFO: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *INFO.get_or_init(|| {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        let status = unsafe { mach_timebase_info(&mut info) };
        if status == 0 && info.numer != 0 && info.denom != 0 {
            Some((info.numer, info.denom))
        } else {
            None
        }
    })
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
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
