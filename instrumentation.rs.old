use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_foundation::NSData;
use objc2_foundation::{NSRange, NSString, NSUInteger};
use objc2_metal::{
    MTLCommandBuffer, MTLCommonCounterSetTimestamp, MTLCounterErrorValue, MTLCounterResultTimestamp, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
};
use rustc_hash::FxHashMap;

use crate::metallic::kernels::matmul::{MatMulBackend, MatMulSample};
use crate::metallic::operation::CommandBuffer;

#[cfg(target_os = "macos")]
use mach2::{
    kern_return::KERN_SUCCESS,
    mach_time::{mach_timebase_info, mach_timebase_info_data_t},
};
#[cfg(target_os = "macos")]
use std::{ptr::NonNull, thread};

/// Handle to a shared latency collector used to instrument fine-grained timing inside
/// the Metal execution context. The collector is populated by `Context` while the
/// inference loops execute and later inspected by higher-level orchestration code.
pub type LatencyCollectorHandle = Rc<RefCell<StepLatencyCollector>>;
pub type MemoryCollectorHandle = Rc<RefCell<StepMemoryCollector>>;

/// Structured description of the matrix dimensions used in a matmul dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Shared recorder handle that allows matmul dispatches to append timing samples
/// once the GPU finishes executing a command buffer.
#[derive(Clone)]
pub struct MatMulSampleRecorder {
    inner: Arc<dyn Fn(MatMulSample) + Send + Sync>,
}

impl MatMulSampleRecorder {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(MatMulSample) + Send + Sync + 'static,
    {
        Self { inner: Arc::new(callback) }
    }

    pub fn record(&self, sample: MatMulSample) {
        (self.inner)(sample);
    }
}

/// Enumerates the sampling strategy required to surround a matmul dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulDispatchKind {
    Compute,
    Blit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulDispatchHandle {
    key: usize,
    index: usize,
}

#[derive(Clone)]
pub struct MatMulDispatchTiming {
    sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    start_index: NSUInteger,
    end_index: NSUInteger,
    kind: MatMulDispatchKind,
}

impl MatMulDispatchTiming {
    pub fn sample_buffer(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &*self.sample_buffer
    }

    pub fn start_index(&self) -> NSUInteger {
        self.start_index
    }

    pub fn end_index(&self) -> NSUInteger {
        self.end_index
    }

    pub fn kind(&self) -> MatMulDispatchKind {
        self.kind
    }
}

pub struct MatMulDispatchRegistration {
    handle: MatMulDispatchHandle,
    timing: Option<MatMulDispatchTiming>,
}

impl MatMulDispatchRegistration {
    pub fn handle(&self) -> MatMulDispatchHandle {
        self.handle
    }

    pub fn timing(&self) -> Option<&MatMulDispatchTiming> {
        self.timing.as_ref()
    }
}

/// Tracks in-flight matmul dispatches so their GPU execution time can be
/// recorded once the surrounding command buffer completes.
#[derive(Clone)]
pub struct MatMulInstrumentation {
    inner: Arc<MatMulInstrumentationInner>,
}

struct MatMulInstrumentationInner {
    pending: Mutex<FxHashMap<usize, PendingMatMul>>,
    #[cfg(target_os = "macos")]
    counter: CounterResources,
}

impl MatMulInstrumentation {
    pub fn new(device: Option<&ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            inner: Arc::new(MatMulInstrumentationInner {
                pending: Mutex::new(FxHashMap::default()),
                #[cfg(target_os = "macos")]
                counter: CounterResources::new(device),
            }),
        }
    }

    fn lock_pending(&self) -> MutexGuard<'_, FxHashMap<usize, PendingMatMul>> {
        self.inner.pending.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn register(
        &self,
        command_buffer: &CommandBuffer,
        backend: MatMulBackend,
        dims: Option<MatmulDims>,
        kind: MatMulDispatchKind,
        recorder: MatMulSampleRecorder,
    ) -> MatMulDispatchRegistration {
        let key = Self::buffer_key(command_buffer);

        {
            let mut pending = self.lock_pending();
            if let Some(entry) = pending.get_mut(&key) {
                let index = entry.next_index();
                let handle = MatMulDispatchHandle { key, index };
                let timing = entry.push_dispatch(handle, backend, dims, self.inner.allocate_timing(command_buffer, kind));
                return MatMulDispatchRegistration { handle, timing };
            }
        }

        self.install_completion(command_buffer, key);

        let mut entry = PendingMatMul::new(recorder);
        let handle = MatMulDispatchHandle {
            key,
            index: entry.next_index(),
        };
        let timing = entry.push_dispatch(handle, backend, dims, self.inner.allocate_timing(command_buffer, kind));
        self.lock_pending().insert(key, entry);

        MatMulDispatchRegistration { handle, timing }
    }

    fn install_completion(&self, command_buffer: &CommandBuffer, key: usize) {
        let instrumentation = self.clone();
        command_buffer.on_completed(move |raw| instrumentation.complete(key, raw));
    }

    fn complete(&self, key: usize, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        let Some(entry) = self.lock_pending().remove(&key) else {
            return;
        };

        let PendingMatMul { recorder, dispatches } = entry;
        let mut resolved_total = Duration::default();
        let mut fallback = Vec::new();

        for dispatch in dispatches {
            let PendingDispatch {
                handle,
                backend,
                dims,
                timing,
            } = dispatch;

            if let Some(timing) = timing.as_ref() {
                if let Some(duration) = self.inner.resolve_duration(timing) {
                    recorder.record(MatMulSample {
                        backend,
                        duration,
                        dims,
                        handle: Some(handle),
                    });
                    resolved_total += duration;
                    continue;
                }
            }

            fallback.push(PendingDispatch {
                handle,
                backend,
                dims,
                timing,
            });
        }

        if fallback.is_empty() {
            return;
        }

        let remaining = Self::command_buffer_gpu_duration(command_buffer).map(|total| total.saturating_sub(resolved_total));
        Self::dispatch_fallback(&recorder, fallback, remaining);
    }

    fn command_buffer_gpu_duration(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Option<Duration> {
        let gpu_start = unsafe { command_buffer.GPUStartTime() };
        let gpu_end = unsafe { command_buffer.GPUEndTime() };

        if !gpu_start.is_finite() || !gpu_end.is_finite() {
            return None;
        }

        let delta = gpu_end - gpu_start;
        if delta <= 0.0 {
            return None;
        }

        Some(Duration::from_secs_f64(delta))
    }

    fn dispatch_fallback(recorder: &MatMulSampleRecorder, dispatches: Vec<PendingDispatch>, remaining: Option<Duration>) {
        let Some(total) = remaining else {
            return;
        };

        if total.is_zero() || dispatches.is_empty() {
            return;
        }

        let per_dispatch = total.as_secs_f64() / dispatches.len() as f64;
        if !per_dispatch.is_finite() || per_dispatch <= 0.0 {
            return;
        }

        let duration = Duration::from_secs_f64(per_dispatch);
        for dispatch in dispatches {
            recorder.record(MatMulSample {
                backend: dispatch.backend,
                duration,
                dims: dispatch.dims,
                handle: Some(dispatch.handle),
            });
        }
    }

    fn buffer_key(command_buffer: &CommandBuffer) -> usize {
        Retained::as_ptr(command_buffer.raw()) as usize
    }
}

impl Default for MatMulInstrumentation {
    fn default() -> Self {
        Self::new(None)
    }
}

struct PendingMatMul {
    recorder: MatMulSampleRecorder,
    dispatches: Vec<PendingDispatch>,
}

impl PendingMatMul {
    fn new(recorder: MatMulSampleRecorder) -> Self {
        Self {
            recorder,
            dispatches: Vec::new(),
        }
    }

    fn next_index(&self) -> usize {
        self.dispatches.len()
    }

    fn push_dispatch(
        &mut self,
        handle: MatMulDispatchHandle,
        backend: MatMulBackend,
        dims: Option<MatmulDims>,
        timing: Option<PendingDispatchTiming>,
    ) -> Option<MatMulDispatchTiming> {
        let timing_for_handle = timing.as_ref().map(PendingDispatchTiming::to_public);
        self.dispatches.push(PendingDispatch {
            handle,
            backend,
            dims,
            timing,
        });
        timing_for_handle
    }
}

struct PendingDispatch {
    handle: MatMulDispatchHandle,
    backend: MatMulBackend,
    dims: Option<MatmulDims>,
    timing: Option<PendingDispatchTiming>,
}

struct PendingDispatchTiming {
    sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    start_index: NSUInteger,
    end_index: NSUInteger,
    kind: MatMulDispatchKind,
}

impl PendingDispatchTiming {
    fn to_public(&self) -> MatMulDispatchTiming {
        MatMulDispatchTiming {
            sample_buffer: self.sample_buffer.clone(),
            start_index: self.start_index,
            end_index: self.end_index,
            kind: self.kind,
        }
    }
}

impl MatMulInstrumentationInner {
    fn allocate_timing(&self, command_buffer: &CommandBuffer, kind: MatMulDispatchKind) -> Option<PendingDispatchTiming> {
        #[cfg(target_os = "macos")]
        {
            self.counter.allocate_timing(command_buffer, kind)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (command_buffer, kind);
            None
        }
    }

    fn resolve_duration(&self, timing: &PendingDispatchTiming) -> Option<Duration> {
        #[cfg(target_os = "macos")]
        {
            self.counter.resolve_duration(timing)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = timing;
            None
        }
    }
}

#[cfg(target_os = "macos")]
struct CounterResources {
    timestamp_counter_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
    timestamp_period: Option<f64>,
    supports_dispatch_sampling: bool,
    supports_blit_sampling: bool,
}

#[cfg(target_os = "macos")]
impl CounterResources {
    fn new(device: Option<&ProtocolObject<dyn MTLDevice>>) -> Self {
        let Some(device) = device else {
            return Self {
                timestamp_counter_set: None,
                timestamp_period: None,
                supports_dispatch_sampling: false,
                supports_blit_sampling: false,
            };
        };

        let counter_set = unsafe { Self::find_timestamp_counter_set(device) };
        let period = Self::gpu_timestamp_period(device);
        let supports_dispatch_sampling = device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary);
        let supports_blit_sampling = device.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary);

        Self {
            timestamp_counter_set: counter_set,
            timestamp_period: period,
            supports_dispatch_sampling,
            supports_blit_sampling,
        }
    }

    fn allocate_timing(&self, command_buffer: &CommandBuffer, kind: MatMulDispatchKind) -> Option<PendingDispatchTiming> {
        let counter_set = self.timestamp_counter_set.as_ref()?;
        match kind {
            MatMulDispatchKind::Compute if !self.supports_dispatch_sampling => return None,
            MatMulDispatchKind::Blit if !self.supports_blit_sampling => return None,
            _ => {}
        }

        let descriptor = unsafe { MTLCounterSampleBufferDescriptor::new() };
        unsafe {
            descriptor.setCounterSet(Some(counter_set.as_ref()));
            descriptor.setSampleCount(2);
        }

        let device = unsafe { command_buffer.raw().device() };
        let sample_buffer = unsafe {
            match device.newCounterSampleBufferWithDescriptor_error(&descriptor) {
                Ok(buffer) => buffer,
                Err(_) => return None,
            }
        };

        Some(PendingDispatchTiming {
            sample_buffer,
            start_index: 0,
            end_index: 1,
            kind,
        })
    }

    fn resolve_duration(&self, timing: &PendingDispatchTiming) -> Option<Duration> {
        let period = self.timestamp_period?;
        let length = timing.end_index - timing.start_index + 1;
        let data: Retained<NSData> = unsafe { timing.sample_buffer.resolveCounterRange(NSRange::new(timing.start_index, length))? };
        let bytes = unsafe { data.as_bytes_unchecked() };
        let stride = core::mem::size_of::<MTLCounterResultTimestamp>();
        if bytes.len() < stride * 2 {
            return None;
        }

        let samples = unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const MTLCounterResultTimestamp, bytes.len() / stride) };

        let start = samples.get(timing.start_index as usize)?.timestamp;
        let end = samples.get(timing.end_index as usize)?.timestamp;
        if start == MTLCounterErrorValue || end == MTLCounterErrorValue || end <= start {
            return None;
        }

        let delta = (end - start) as f64 * period;
        if delta <= 0.0 {
            return None;
        }

        Some(Duration::from_secs_f64(delta))
    }

    unsafe fn find_timestamp_counter_set(device: &ProtocolObject<dyn MTLDevice>) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
        let sets = unsafe { device.counterSets()? };
        let desired: &NSString = unsafe { MTLCommonCounterSetTimestamp };
        let count = sets.count() as usize;
        for idx in 0..count {
            let set = sets.objectAtIndex(idx as NSUInteger);
            let name = unsafe { set.name() };
            let matches = unsafe { name.isEqualToString(desired) };
            if matches {
                return Some(set);
            }
        }
        None
    }

    fn gpu_timestamp_period(device: &ProtocolObject<dyn MTLDevice>) -> Option<f64> {
        let mut cpu_start: u64 = 0;
        let mut gpu_start: u64 = 0;
        unsafe {
            device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_start), NonNull::from(&mut gpu_start));
        }

        thread::sleep(Duration::from_micros(200));

        let mut cpu_end: u64 = 0;
        let mut gpu_end: u64 = 0;
        unsafe {
            device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_end), NonNull::from(&mut gpu_end));
        }

        let cpu_delta = cpu_end.saturating_sub(cpu_start);
        let gpu_delta = gpu_end.saturating_sub(gpu_start);
        if cpu_delta == 0 || gpu_delta == 0 {
            return None;
        }

        let mut info = mach_timebase_info_data_t { numer: 0, denom: 0 };
        let status = unsafe { mach_timebase_info(&mut info) };
        if status != KERN_SUCCESS || info.denom == 0 {
            return None;
        }

        let cpu_delta_ns = (cpu_delta as u128 * info.numer as u128) / info.denom as u128;
        Some(cpu_delta_ns as f64 / 1e9 / gpu_delta as f64)
    }
}

#[cfg(not(target_os = "macos"))]
struct CounterResources;

#[cfg(not(target_os = "macos"))]
impl CounterResources {
    fn new(_device: Option<&ProtocolObject<dyn MTLDevice>>) -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn fallback_splits_remaining_time_evenly() {
        let samples: Arc<Mutex<Vec<MatMulSample>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = MatMulSampleRecorder::new({
            let samples = Arc::clone(&samples);
            move |sample| {
                if let Ok(mut guard) = samples.lock() {
                    guard.push(sample);
                }
            }
        });

        let dispatches = vec![
            PendingDispatch {
                handle: MatMulDispatchHandle { key: 0, index: 0 },
                backend: MatMulBackend::Mps,
                dims: None,
                timing: None,
            },
            PendingDispatch {
                handle: MatMulDispatchHandle { key: 0, index: 1 },
                backend: MatMulBackend::Mps,
                dims: None,
                timing: None,
            },
        ];

        MatMulInstrumentation::dispatch_fallback(&recorder, dispatches, Some(Duration::from_millis(30)));

        let recorded = samples.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        for sample in recorded.iter() {
            assert_eq!(sample.backend, MatMulBackend::Mps);
            assert_eq!(sample.duration, Duration::from_millis(15));
            assert!(sample.dims.is_none());
            assert!(sample.handle.is_some());
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

    /// Reset internal state without dropping any allocated storage.
    pub fn reset(&mut self) {
        self.forward_step = None;
        for block in &mut self.block_durations {
            block.reset();
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
        let mut snapshot = StepLatencySnapshot::empty(self.block_durations.len());
        self.snapshot_into(&mut snapshot);
        snapshot
    }

    /// Populate the provided snapshot with the most recent measurements.
    pub fn snapshot_into(&self, snapshot: &mut StepLatencySnapshot) {
        snapshot.forward_step = self.forward_step.unwrap_or_default();
        snapshot.ensure_block_capacity(self.block_durations.len());
        for (index, block) in self.block_durations.iter().enumerate() {
            block.snapshot_into(&mut snapshot.blocks[index]);
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

    pub fn ensure_block_capacity(&mut self, block_count: usize) {
        if self.blocks.len() < block_count {
            self.blocks.resize_with(block_count, BlockLatencySnapshot::default);
        } else if self.blocks.len() > block_count {
            self.blocks.truncate(block_count);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlockLatencyEntry {
    total: Option<Duration>,
    phases: Vec<BlockPhaseEntry>,
}

impl BlockLatencyEntry {
    fn reset(&mut self) {
        self.total = None;
        self.phases.clear();
    }

    fn snapshot_into(&self, snapshot: &mut BlockLatencySnapshot) {
        snapshot.total = self.total.unwrap_or_default();
        snapshot.ensure_phase_capacity(self.phases.len());
        for (index, phase) in self.phases.iter().enumerate() {
            let slot = &mut snapshot.phases[index];
            slot.label.clear();
            slot.label.push_str(&phase.label);
            slot.duration = phase.duration;
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

impl BlockLatencySnapshot {
    fn ensure_phase_capacity(&mut self, phase_count: usize) {
        if self.phases.len() < phase_count {
            self.phases.resize_with(phase_count, BlockPhaseSnapshot::default);
        } else if self.phases.len() > phase_count {
            self.phases.truncate(phase_count);
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockPhaseSnapshot {
    pub label: String,
    pub duration: Duration,
}

impl Default for BlockPhaseSnapshot {
    fn default() -> Self {
        Self {
            label: String::new(),
            duration: Duration::default(),
        }
    }
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

    pub fn into_owned(self) -> MemoryEvent<'static> {
        match self {
            MemoryEvent::ForwardStart => MemoryEvent::ForwardStart,
            MemoryEvent::ForwardSample => MemoryEvent::ForwardSample,
            MemoryEvent::BlockStart { index } => MemoryEvent::BlockStart { index },
            MemoryEvent::BlockEnd { index } => MemoryEvent::BlockEnd { index },
            MemoryEvent::BlockPhase { index, label } => MemoryEvent::BlockPhase {
                index,
                label: Cow::Owned(label.into_owned()),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemorySample {
    pub event: MemoryEvent<'static>,
    pub usage: MemoryUsage,
}

pub type MemorySampleSender = Sender<MemorySample>;

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
    sample_tx: MemorySampleSender,
    sample_rx: Receiver<MemorySample>,
}

impl StepMemoryCollector {
    pub fn new(block_count: usize) -> Self {
        let (sample_tx, sample_rx) = mpsc::channel();
        Self {
            forward: MemoryScopeEntry::default(),
            blocks: vec![BlockMemoryEntry::default(); block_count],
            sample_tx,
            sample_rx,
        }
    }

    pub fn sample_sender(&self) -> MemorySampleSender {
        self.sample_tx.clone()
    }

    pub fn record(&mut self, event: MemoryEvent<'_>, usage: MemoryUsage) {
        self.apply_sample(MemorySample {
            event: event.into_owned(),
            usage,
        });
    }

    pub fn reset(&mut self) {
        self.forward.reset();
        for block in &mut self.blocks {
            block.reset();
        }
        while let Ok(_) = self.sample_rx.try_recv() {}
    }

    pub fn drain_pending(&mut self) {
        while let Ok(sample) = self.sample_rx.try_recv() {
            self.apply_sample(sample);
        }
    }

    pub fn snapshot(&mut self) -> StepMemorySnapshot {
        let mut snapshot = StepMemorySnapshot::empty(self.blocks.len());
        self.snapshot_into(&mut snapshot);
        snapshot
    }

    pub fn snapshot_into(&mut self, snapshot: &mut StepMemorySnapshot) {
        self.drain_pending();
        self.forward.snapshot_into(&mut snapshot.forward);
        snapshot.ensure_block_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            block.snapshot_into(&mut snapshot.blocks[index]);
        }
    }

    fn apply_sample(&mut self, sample: MemorySample) {
        match sample.event {
            MemoryEvent::ForwardStart => {
                self.forward.record_baseline(sample.usage);
                for block in &mut self.blocks {
                    block.reset();
                }
            }
            MemoryEvent::ForwardSample => {
                self.forward.record_sample(sample.usage);
            }
            MemoryEvent::BlockStart { index } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_baseline(sample.usage);
                }
                self.forward.record_sample(sample.usage);
            }
            MemoryEvent::BlockEnd { index } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_sample(sample.usage);
                }
                self.forward.record_sample(sample.usage);
            }
            MemoryEvent::BlockPhase { index, label } => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.record_phase(label.into_owned(), sample.usage);
                }
                self.forward.record_sample(sample.usage);
            }
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
    fn reset(&mut self) {
        self.baseline = None;
        self.last = None;
        self.peak_pool_delta = 0;
        self.peak_kv_delta = 0;
        self.peak_kv_cache_delta = 0;
    }

    fn record_baseline(&mut self, usage: MemoryUsage) {
        self.reset();
        self.baseline = Some(usage);
        self.last = Some(usage);
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
        let mut snapshot = MemoryScopeSnapshot::default();
        self.snapshot_into(&mut snapshot);
        snapshot
    }

    fn snapshot_into(&self, snapshot: &mut MemoryScopeSnapshot) {
        let (current_pool_delta, current_kv_delta, current_kv_cache_delta) = if let (Some(base), Some(last)) = (self.baseline, self.last) {
            let delta = last.delta_from(base);
            (delta.pool_used, delta.kv_used, delta.kv_cache_bytes)
        } else {
            (0, 0, 0)
        };

        snapshot.baseline = self.baseline;
        snapshot.last = self.last;
        snapshot.current_pool_delta = current_pool_delta;
        snapshot.current_kv_delta = current_kv_delta;
        snapshot.current_kv_cache_delta = current_kv_cache_delta;
        snapshot.peak_pool_delta = self.peak_pool_delta;
        snapshot.peak_kv_delta = self.peak_kv_delta;
        snapshot.peak_kv_cache_delta = self.peak_kv_cache_delta;
    }
}

#[derive(Clone, Debug, Default)]
struct BlockMemoryEntry {
    scope: MemoryScopeEntry,
    phases: Vec<MemoryPhaseEntry>,
}

impl BlockMemoryEntry {
    fn reset(&mut self) {
        self.scope.reset();
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

    fn snapshot_into(&self, snapshot: &mut BlockMemorySnapshot) {
        let mut scope_snapshot = MemoryScopeSnapshot::default();
        self.scope.snapshot_into(&mut scope_snapshot);
        snapshot.apply_scope(&scope_snapshot);
        snapshot.ensure_phase_capacity(self.phases.len());
        for (index, phase) in self.phases.iter().enumerate() {
            phase.snapshot_into(&mut snapshot.phases[index]);
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
    fn snapshot_into(&self, snapshot: &mut MemoryPhaseSnapshot) {
        snapshot.label.clear();
        snapshot.label.push_str(&self.label);
        snapshot.current_pool_delta = self.current_pool_delta;
        snapshot.current_kv_delta = self.current_kv_delta;
        snapshot.current_kv_cache_delta = self.current_kv_cache_delta;
        snapshot.peak_pool_delta = self.peak_pool_delta;
        snapshot.peak_kv_delta = self.peak_kv_delta;
        snapshot.peak_kv_cache_delta = self.peak_kv_cache_delta;
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

impl BlockMemorySnapshot {
    fn apply_scope(&mut self, scope: &MemoryScopeSnapshot) {
        self.baseline = scope.baseline;
        self.last = scope.last;
        self.current_pool_delta = scope.current_pool_delta;
        self.current_kv_delta = scope.current_kv_delta;
        self.current_kv_cache_delta = scope.current_kv_cache_delta;
        self.peak_pool_delta = scope.peak_pool_delta;
        self.peak_kv_delta = scope.peak_kv_delta;
        self.peak_kv_cache_delta = scope.peak_kv_cache_delta;
    }

    fn ensure_phase_capacity(&mut self, phase_count: usize) {
        if self.phases.len() < phase_count {
            self.phases.resize_with(phase_count, MemoryPhaseSnapshot::default);
        } else if self.phases.len() > phase_count {
            self.phases.truncate(phase_count);
        }
    }
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

impl Default for MemoryPhaseSnapshot {
    fn default() -> Self {
        Self {
            label: String::new(),
            current_pool_delta: 0,
            current_kv_delta: 0,
            current_kv_cache_delta: 0,
            peak_pool_delta: 0,
            peak_kv_delta: 0,
            peak_kv_cache_delta: 0,
        }
    }
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

    pub fn ensure_block_capacity(&mut self, block_count: usize) {
        if self.blocks.len() < block_count {
            self.blocks.resize_with(block_count, BlockMemorySnapshot::default);
        } else if self.blocks.len() > block_count {
            self.blocks.truncate(block_count);
        }
    }
}

pub fn new_memory_collector(block_count: usize) -> MemoryCollectorHandle {
    Rc::new(RefCell::new(StepMemoryCollector::new(block_count)))
}
