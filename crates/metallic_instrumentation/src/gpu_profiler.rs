//! GPU command-buffer profiler emitting per-operation metrics.

use objc2::exception::catch;
use objc2::rc::Retained;
use objc2::runtime::{Bool, NSObjectProtocol, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_foundation::{NSRange, NSString, NSUInteger};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLCounterErrorValue,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
};
use std::env;
use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use tracing::Dispatch;
use tracing::{self, dispatcher};

use mach2::{
    kern_return::KERN_SUCCESS,
    mach_time::{mach_timebase_info, mach_timebase_info_data_t},
};

use crate::event::MetricEvent;
use crate::record_metric;

pub trait ProfiledCommandBuffer {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    fn on_completed(&self, handler: Box<dyn FnOnce(&ProtocolObject<dyn MTLCommandBuffer>, Option<std::time::Instant>) + Send + 'static>);
}

#[derive(Clone)]
pub struct GpuProfiler {
    // Keep the profiler state alive for the lifetime of this handle so scopes
    // may continue to resolve even if the handle is dropped early.
    _state: Arc<GpuProfilerState>,
}

pub struct GpuProfilerScope {
    inner: Option<GpuProfilerScopeInner>,
}

impl GpuProfilerScope {
    pub fn finish(mut self) {
        if let Some(inner) = self.inner.take() {
            inner.complete();
        }
    }
}

impl Drop for GpuProfilerScope {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            inner.complete();
        }
    }
}

struct GpuProfilerScopeInner {
    state: Arc<GpuProfilerState>,
    timing: Option<PendingTiming>,
    encoder: Option<EncoderHandle>,
    op_name: String,
    backend: String,
    cpu_start: Instant,
}

impl GpuProfilerScopeInner {
    fn begin_timing(&mut self) {
        // Only sample counters if feature is explicitly enabled
        let enabled = env::var("METALLIC_ENABLE_GPU_COUNTERS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
            return;
        }

        let Some(timing) = self.timing.as_ref() else {
            return;
        };
        let Some(encoder) = self.encoder.as_ref() else {
            return;
        };

        let recorded = encoder.sample_counters(&*timing.sample_buffer, timing.start_index, Bool::NO);
        if !recorded {
            tracing::debug!(target: "instrument", "Failed to record GPU start sample for {}; falling back to command buffer timing", self.op_name);
            self.timing = None;
        }
    }

    fn finish_timing(&mut self) {
        let enabled = env::var("METALLIC_ENABLE_GPU_COUNTERS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
            return;
        }

        let Some(timing) = self.timing.as_ref() else {
            return;
        };
        let Some(encoder) = self.encoder.as_ref() else {
            return;
        };

        let recorded = encoder.sample_counters(&*timing.sample_buffer, timing.end_index, Bool::YES);
        if !recorded {
            tracing::debug!(target: "instrument", "Failed to record GPU end sample for {}; falling back to command buffer timing", self.op_name);
            self.timing = None;
        }
    }

    fn complete(mut self) {
        self.finish_timing();

        let record = GpuOpRecord {
            op_name: self.op_name,
            backend: self.backend,
            timing: if let Some(timing) = self.timing {
                // Store the timing information so it can be resolved later when the command buffer completes
                GpuTiming::Counters {
                    sample_buffer: timing.sample_buffer,
                    start_index: timing.start_index,
                    end_index: timing.end_index,
                }
            } else {
                GpuTiming::None
            },
            cpu_start: self.cpu_start,
            cpu_end: Instant::now(),
        };

        self.state.push_record(record);
    }
}

struct GpuProfilerState {
    key: usize,
    counter: CounterResources,
    dispatch: Dispatch,
    records: Mutex<Vec<GpuOpRecord>>,
    sequence: Mutex<u64>,
}

impl GpuProfilerState {
    fn new(key: usize, counter: CounterResources, dispatch: Dispatch) -> Self {
        Self {
            key,
            counter,
            dispatch,
            records: Mutex::new(Vec::new()),
            sequence: Mutex::new(0),
        }
    }

    fn next_sequence(&self) -> u64 {
        let mut guard = self.sequence.lock().expect("sequence mutex poisoned");
        let value = *guard;
        *guard = guard.saturating_add(1);
        value
    }

    fn push_record(&self, record: GpuOpRecord) {
        let mut guard = self.records.lock().expect("record mutex poisoned");
        guard.push(record);
    }

    fn take_records(&self) -> Vec<GpuOpRecord> {
        let mut guard = self.records.lock().expect("record mutex poisoned");
        guard.drain(..).collect()
    }

    fn process_completion(&self) {
        dispatcher::with_default(&self.dispatch, || {
            // Get records but process them with fallback approach that prioritizes counter resolution
            let records = self.take_records();
            if records.is_empty() {
                registry().lock().expect("registry mutex poisoned").remove(&self.key);
                return;
            }

            // First, try to resolve individual GPU timing from counter buffers
            let mut fallback_records = Vec::new();

            for record in records {
                if let Some(duration) = self.counter.resolve_duration(&record) {
                    let duration_us = (duration.as_secs_f64() * 1e6).max(1.0).round() as u64;
                    record_metric!(MetricEvent::GpuOpCompleted {
                        op_name: record.op_name,
                        backend: record.backend,
                        duration_us,
                    });
                } else {
                    // This record needs to use fallback timing
                    fallback_records.push(record);
                }
            }

            // For records that couldn't get individual timing, distribute the remaining
            // command buffer time evenly across them using GPUStartTime/GPUEndTime.
            if !fallback_records.is_empty() {
                for record in fallback_records {
                    let cpu_duration = record.cpu_start.elapsed();
                    let duration = if cpu_duration.is_zero() {
                        Duration::from_micros(1)
                    } else {
                        cpu_duration
                    };
                    let duration_us = (duration.as_secs_f64() * 1e6).max(1.0).round() as u64;
                    record_metric!(MetricEvent::GpuOpCompleted {
                        op_name: record.op_name,
                        backend: record.backend,
                        duration_us,
                    });
                }
            }
            // Already handled fallback_records above; nothing else to emit here.

            registry().lock().expect("registry mutex poisoned").remove(&self.key);
        });
    }

    fn process_completion_with_buffer(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        host_commit: Option<std::time::Instant>,
    ) {
        let records = self.take_records();
        if records.is_empty() {
            registry().lock().expect("registry mutex poisoned").remove(&self.key);
            return;
        }

        let mut resolved_total = Duration::from_micros(0);
        let mut fallback_records = Vec::new();

        for record in records {
            if let Some(duration) = self.counter.resolve_duration(&record) {
                let duration_us = (duration.as_secs_f64() * 1e6).max(1.0).round() as u64;
                record_metric!(MetricEvent::GpuOpCompleted {
                    op_name: record.op_name,
                    backend: record.backend,
                    duration_us,
                });
                resolved_total = resolved_total.saturating_add(duration);
            } else {
                fallback_records.push(record);
            }
        }

        if !fallback_records.is_empty() {
            // Determine remaining GPU time if available
            let gpu_start = command_buffer.GPUStartTime();
            let gpu_end = command_buffer.GPUEndTime();
            let gpu_total = if gpu_start.is_finite() && gpu_end.is_finite() && gpu_end > gpu_start {
                Some(Duration::from_secs_f64(gpu_end - gpu_start))
            } else {
                None
            };
            let host_total = host_commit.map(|hc| hc.elapsed());
            let total = gpu_total.or(host_total);
            let remaining = total.and_then(|t| t.checked_sub(resolved_total));

            // Build weights from CPU spans captured around encode
            let mut weights = Vec::with_capacity(fallback_records.len());
            let mut weight_sum = Duration::from_micros(0);
            for r in &fallback_records {
                let w = r.cpu_end.saturating_duration_since(r.cpu_start);
                let w = if w.is_zero() { Duration::from_micros(1) } else { w };
                weight_sum = weight_sum.saturating_add(w);
                weights.push(w);
            }

            if let Some(rem) = remaining {
                // Distribute remaining GPU time proportionally by CPU weights
                let rem_secs = rem.as_secs_f64();
                let sum_secs = weight_sum.as_secs_f64().max(1e-6);
                for (record, w) in fallback_records.into_iter().zip(weights.into_iter()) {
                    let frac = (w.as_secs_f64() / sum_secs).clamp(0.0, 1.0);
                    let dur = Duration::from_secs_f64((rem_secs * frac).max(1e-6));
                    let duration_us = (dur.as_secs_f64() * 1e6).max(1.0).round() as u64;
                    record_metric!(MetricEvent::GpuOpCompleted {
                        op_name: record.op_name,
                        backend: record.backend,
                        duration_us,
                    });
                }
            } else {
                // No GPU timing available; report CPU spans directly for relative visibility
                for (record, w) in fallback_records.into_iter().zip(weights.into_iter()) {
                    let duration_us = (w.as_secs_f64() * 1e6).max(1.0).round() as u64;
                    record_metric!(MetricEvent::GpuOpCompleted {
                        op_name: record.op_name,
                        backend: record.backend,
                        duration_us,
                    });
                }
            }
        }

        registry().lock().expect("registry mutex poisoned").remove(&self.key);
    }

    fn process_completion_inner_fallback(&self) {
        let records = self.take_records();
        if records.is_empty() {
            registry().lock().expect("registry mutex poisoned").remove(&self.key);
            return;
        }

        // Use CPU-based timing fallback when command buffer isn't available
        for record in records {
            let duration = record.cpu_start.elapsed();
            let mut duration = duration;
            if duration.is_zero() {
                duration = Duration::from_micros(1);
            }

            let duration_us = (duration.as_secs_f64() * 1e6).max(1.0).round() as u64;
            record_metric!(MetricEvent::GpuOpCompleted {
                op_name: record.op_name,
                backend: record.backend,
                duration_us,
            });
        }

        registry().lock().expect("registry mutex poisoned").remove(&self.key);
    }
}

#[derive(Clone)]
struct GpuOpRecord {
    op_name: String,
    backend: String,
    timing: GpuTiming,
    cpu_start: Instant,
    cpu_end: Instant,
}

#[derive(Clone)]
enum GpuTiming {
    Counters {
        sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
        start_index: NSUInteger,
        end_index: NSUInteger,
    },
    None,
}

// SAFETY: `MTLCounterSampleBuffer` is documented by Apple as thread-safe, and we only
// issue immutable method calls (`resolveCounterRange`) from the completion handler.
// Objective-C reference counting already enforces correct lifetimes across threads.
unsafe impl Send for GpuTiming {}

// SAFETY: See `Send` rationale above; sharing immutable references to the retained
// counter sample buffer between threads is safe.
unsafe impl Sync for GpuTiming {}

// SAFETY: `GpuOpRecord` only contains owned data and `GpuTiming`, which is safe to
// share across threads as documented above.
unsafe impl Send for GpuOpRecord {}

// SAFETY: See `Send` rationale.
unsafe impl Sync for GpuOpRecord {}

#[derive(Clone)]
enum EncoderHandle {
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

impl EncoderHandle {
    fn sample_counters(&self, sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>, index: NSUInteger, barrier: Bool) -> bool {
        let mut encoder_kind = "unknown";

        let result = unsafe {
            catch(AssertUnwindSafe(|| match self {
                EncoderHandle::Compute(encoder) => {
                    encoder_kind = "compute";
                    if encoder.respondsToSelector(sel!(sampleCountersInBuffer:atSampleIndex:withBarrier:)) {
                        let _: () = msg_send![
                            &**encoder,
                            sampleCountersInBuffer: sample_buffer,
                            atSampleIndex: index,
                            withBarrier: barrier
                        ];
                        Ok(())
                    } else {
                        Err(())
                    }
                }
                EncoderHandle::Blit(encoder) => {
                    encoder_kind = "blit";
                    if encoder.respondsToSelector(sel!(sampleCountersInBuffer:atSampleIndex:withBarrier:)) {
                        let _: () = msg_send![
                            &**encoder,
                            sampleCountersInBuffer: sample_buffer,
                            atSampleIndex: index,
                            withBarrier: barrier
                        ];
                        Ok(())
                    } else {
                        Err(())
                    }
                }
            }))
        };

        match result {
            Ok(Ok(())) => true,
            _ => {
                tracing::debug!(
                    target: "instrument",
                    "Failed to sample {} encoder counters at index {} (with_barrier: {}); falling back to CPU timing",
                    encoder_kind,
                    index,
                    barrier.as_raw()
                );
                false
            }
        }
    }
}

fn registry() -> &'static Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>> {
    static REGISTRY: OnceLock<Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn buffer_key<C: ProfiledCommandBuffer + ?Sized>(command_buffer: &C) -> usize {
    Retained::as_ptr(command_buffer.raw()) as usize
}

impl GpuProfiler {
    pub fn attach<C: ProfiledCommandBuffer + ?Sized>(command_buffer: &C) -> Option<Self> {
        let key = buffer_key(command_buffer);
        let raw = command_buffer.raw();
        let profiling_enabled = enable_command_buffer_profiling(raw);
        if !profiling_enabled {
            log_profiling_unavailable_once();
        }
        let device = raw.device();

        // Create counter resources even if we won't use counter sampling
        // This allows us to use other features if available
        let counter = CounterResources::new(Some(device.as_ref()));

        let dispatch = dispatcher::get_default(|dispatch| dispatch.clone());
        let state = Arc::new(GpuProfilerState::new(key, counter, dispatch));

        registry()
            .lock()
            .expect("registry mutex poisoned")
            .insert(key, Arc::downgrade(&state));

        let completion_state = Arc::clone(&state);
        command_buffer.on_completed(Box::new(move |cb, host_commit| {
            completion_state.process_completion_with_buffer(cb, host_commit);
        }));

        Some(Self { _state: state })
    }

    fn scope_for_operation(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        state: Arc<GpuProfilerState>,
        encoder: Option<EncoderHandle>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let timing = if let Some(ref encoder_handle) = encoder {
            // Only allocate counters when enabled
            let enabled = env::var("METALLIC_ENABLE_GPU_COUNTERS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !enabled {
                None
            } else {
                match state.counter.allocate_timing(command_buffer, encoder_handle) {
                    Some(pending) => Some(pending),
                    None => {
                        tracing::debug!(target: "instrument", "Could not allocate timing resources; falling back to command buffer timing");
                        None
                    }
                }
            }
        } else {
            None
        };

        let mut inner = GpuProfilerScopeInner {
            state,
            timing,
            encoder,
            op_name,
            backend,
            cpu_start: Instant::now(),
        };
        inner.begin_timing();

        Some(GpuProfilerScope { inner: Some(inner) })
    }

    fn from_command_buffer(command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Option<Arc<GpuProfilerState>> {
        let key = Retained::as_ptr(command_buffer) as usize;
        registry()
            .lock()
            .expect("registry mutex poisoned")
            .get(&key)
            .and_then(|weak| weak.upgrade())
    }

    pub fn profile_compute(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_operation(
            command_buffer,
            state,
            Some(EncoderHandle::Compute(encoder.clone())),
            format!("{op_name}#{sequence}"),
            backend,
        )
    }

    pub fn profile_blit(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: &Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_operation(
            command_buffer,
            state,
            Some(EncoderHandle::Blit(encoder.clone())),
            format!("{op_name}#{sequence}"),
            backend,
        )
    }

    pub fn profile_command_buffer(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_operation(command_buffer, state, None, format!("{op_name}#{sequence}"), backend)
    }
}

fn enable_command_buffer_profiling(command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> bool {
    unsafe {
        if !command_buffer.respondsToSelector(sel!(setProfilingEnabled:)) {
            return false;
        }

        match catch(AssertUnwindSafe(|| {
            let _: () = msg_send![&**command_buffer, setProfilingEnabled: Bool::YES];
        })) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

fn log_profiling_unavailable_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        tracing::debug!(
            target: "instrument",
            "Metal command buffer profiling is unavailable; GPU timings will fall back to CPU estimates"
        );
    });
}

struct CounterResources {
    timestamp_counter_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
    supports_stage_sampling: bool,
    supports_dispatch_sampling: bool,
    supports_blit_sampling: bool,
    timestamp_period: Option<f64>,
}

// SAFETY: `MTLCounterSet` handles are immutable configuration objects that Metal
// allows to be shared freely across threads. We never mutate the retained object
// after construction, so moving the wrapper between threads is sound.
unsafe impl Send for CounterResources {}

// SAFETY: Same justification as `Send` — the retained counter set is immutable and
// may be safely observed from multiple threads.
unsafe impl Sync for CounterResources {}

struct PendingTiming {
    sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    start_index: NSUInteger,
    end_index: NSUInteger,
}

impl CounterResources {
    fn new(device: Option<&ProtocolObject<dyn MTLDevice>>) -> Self {
        let mut resources = Self {
            timestamp_counter_set: None,
            supports_stage_sampling: false,
            supports_dispatch_sampling: false,
            supports_blit_sampling: false,
            timestamp_period: None,
        };

        if let Some(device) = device {
            resources.timestamp_counter_set = Self::find_timestamp_counter_set(device);
            resources.supports_stage_sampling = device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary);
            resources.supports_dispatch_sampling = device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary);
            resources.supports_blit_sampling = device.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary);
            resources.timestamp_period = Self::gpu_timestamp_period(device);
        }

        resources
    }

    fn allocate_timing(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: &EncoderHandle,
    ) -> Option<PendingTiming> {
        // Don't attempt to allocate timing if the device doesn't support the required sampling
        match encoder {
            EncoderHandle::Compute(_) if !self.supports_dispatch_sampling && !self.supports_stage_sampling => {
                tracing::debug!(
                    target: "instrument",
                    "Compute encoder sampling not supported on this device; falling back to CPU timing"
                );
                return None;
            }
            EncoderHandle::Blit(_) if !self.supports_blit_sampling => {
                tracing::debug!(
                    target: "instrument",
                    "Blit encoder sampling not supported on this device; falling back to CPU timing"
                );
                return None;
            }
            _ => {}
        }

        let counter_set = self.timestamp_counter_set.as_ref()?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        unsafe {
            descriptor.setCounterSet(Some(counter_set.as_ref()));
            descriptor.setSampleCount(2);
        }

        let device = command_buffer.device();
        let sample_buffer = match device.newCounterSampleBufferWithDescriptor_error(&descriptor) {
            Ok(buffer) => buffer,
            Err(error) => {
                tracing::debug!(
                    target: "instrument",
                    "Failed to create counter sample buffer: {:?}",
                    error
                );
                return None;
            }
        };

        Some(PendingTiming {
            sample_buffer,
            start_index: 0,
            end_index: 1,
        })
    }

    fn resolve_duration(&self, record: &GpuOpRecord) -> Option<Duration> {
        let GpuTiming::Counters {
            sample_buffer,
            start_index,
            end_index,
        } = &record.timing
        else {
            return None;
        };

        // Continue processing - we assume sample_buffer is valid since it's passed as a reference

        let period = self.timestamp_period?;
        let length = end_index - start_index + 1;

        // Safety check on length to prevent potential issues
        if *end_index < *start_index {
            tracing::debug!(
                target: "instrument",
                "Invalid index range when resolving duration; falling back to CPU timing"
            );
            return None;
        }

        let result = catch(AssertUnwindSafe(|| unsafe {
            sample_buffer.resolveCounterRange(NSRange::new(*start_index, length))
        }));

        let data = match result {
            Ok(Some(resolved_data)) => resolved_data,
            Ok(None) | Err(_) => {
                tracing::debug!(
                    target: "instrument",
                    "Failed to resolve counter range; falling back to CPU timing"
                );
                return None;
            }
        };

        // Safety check: make sure data has valid content
        if data.length() == 0 {
            tracing::debug!(
                target: "instrument",
                "Resolved counter data has zero length; falling back to CPU timing"
            );
            return None;
        }

        let bytes = unsafe { data.as_bytes_unchecked() };
        if bytes.len() < core::mem::size_of::<MTLCounterResultTimestamp>() * 2 {
            tracing::debug!(
                target: "instrument",
                "Insufficient counter data bytes; falling back to CPU timing"
            );
            return None;
        }

        let samples = unsafe {
            core::slice::from_raw_parts(
                bytes.as_ptr() as *const MTLCounterResultTimestamp,
                bytes.len() / core::mem::size_of::<MTLCounterResultTimestamp>(),
            )
        };

        let start_index: usize = *start_index;
        let end_index: usize = *end_index;

        // Bounds checking for array access
        let start = samples.get(start_index)?.timestamp;
        let end = samples.get(end_index)?.timestamp;

        if start == MTLCounterErrorValue || end == MTLCounterErrorValue || end <= start {
            tracing::debug!(
                target: "instrument",
                "Invalid counter values detected; falling back to CPU timing"
            );
            return None;
        }

        let delta = (end - start) as f64 * period;
        if delta <= 0.0 || !delta.is_finite() {
            tracing::debug!(
                target: "instrument",
                "Invalid counter delta calculated; falling back to CPU timing"
            );
            return None;
        }

        Some(Duration::from_secs_f64(delta))
    }

    fn find_timestamp_counter_set(device: &ProtocolObject<dyn MTLDevice>) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
        let sets = device.counterSets()?;

        // No direct null check available for NSArray, proceed with count check
        let count = sets.count();
        if count == 0 {
            tracing::debug!(
                target: "instrument",
                "No counter sets available"
            );
            return None;
        }

        let desired: &NSString = unsafe { MTLCommonCounterSetTimestamp };
        let count = sets.count();

        // Safety check: avoid potential overflow in count
        if count > 1000 {
            // Arbitrary reasonable limit
            tracing::warn!(
                target: "instrument",
                "Unusually high number of counter sets ({}) - limiting search", count
            );
        }

        let limit = std::cmp::min(count, 1000);
        for idx in 0..limit {
            let set = sets.objectAtIndex(idx as NSUInteger);

            // Note: No direct null check available for set object in this context
            // We'll handle potential exceptions instead

            let result = unsafe { catch(AssertUnwindSafe(|| set.name())) };

            match result {
                Ok(name) => {
                    if name.isEqualToString(desired) {
                        return Some(set);
                    }
                }
                Err(_) => {
                    tracing::debug!(
                        target: "instrument",
                        "Failed to get name for counter set at index {}", idx
                    );
                }
            }
        }
        None
    }

    fn gpu_timestamp_period(device: &ProtocolObject<dyn MTLDevice>) -> Option<f64> {
        let mut info = mach_timebase_info_data_t { numer: 0, denom: 0 };
        if unsafe { mach_timebase_info(&mut info) } != KERN_SUCCESS || info.denom == 0 {
            tracing::debug!(
                target: "instrument",
                "Could not get mach timebase info; GPU timing may be inaccurate"
            );
            return None;
        }

        let mut sleep = Duration::from_micros(200);
        for i in 0..5 {
            let mut cpu_start: u64 = 0;
            let mut gpu_start: u64 = 0;

            // Exception handling for timestamp sampling
            let result = unsafe {
                catch(AssertUnwindSafe(|| {
                    device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_start), NonNull::from(&mut gpu_start));
                }))
            };

            if result.is_err() {
                tracing::debug!(
                    target: "instrument",
                    "Failed to sample GPU timestamps on attempt {}; falling back to CPU timing", i
                );
                continue;
            }

            std::thread::sleep(sleep);

            let mut cpu_end: u64 = 0;
            let mut gpu_end: u64 = 0;

            let result = unsafe {
                catch(AssertUnwindSafe(|| {
                    device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_end), NonNull::from(&mut gpu_end));
                }))
            };

            if result.is_err() {
                tracing::debug!(
                    target: "instrument",
                    "Failed to sample GPU timestamps on second call in attempt {}; falling back to CPU timing", i
                );
                continue;
            }

            let cpu_delta = cpu_end.saturating_sub(cpu_start);
            let gpu_delta = gpu_end.saturating_sub(gpu_start);
            if cpu_delta != 0 && gpu_delta != 0 {
                let cpu_delta_ns = (cpu_delta as u128 * info.numer as u128) / info.denom as u128;
                let period = cpu_delta_ns as f64 / 1e9 / gpu_delta as f64;
                if period.is_finite() && period > 0.0 && period.is_normal() {
                    return Some(period);
                }
            }

            sleep = sleep.saturating_mul(2);
        }

        tracing::debug!(
            target: "instrument",
            "Could not determine GPU timestamp period after 5 attempts; falling back to CPU timing"
        );
        None
    }
}
