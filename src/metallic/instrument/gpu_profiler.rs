//! GPU command-buffer profiler emitting per-operation metrics.

use crate::metallic::instrument::event::MetricEvent;
use crate::metallic::operation::CommandBuffer;
use crate::record_metric;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{Bool, ProtocolObject};
use objc2_foundation::{NSData, NSRange, NSString, NSUInteger};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLCounterErrorValue,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use tracing::Dispatch;
use tracing::{self, dispatcher};

use mach2::{
    kern_return::KERN_SUCCESS,
    mach_time::{mach_timebase_info, mach_timebase_info_data_t},
};

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
    encoder: EncoderHandle,
    op_name: String,
    backend: String,
    cpu_start: Instant,
}

impl GpuProfilerScopeInner {
    fn complete(mut self) {
        let cpu_duration = self.cpu_start.elapsed();
        let mut record = GpuOpRecord {
            op_name: self.op_name,
            backend: self.backend,
            timing: GpuTiming::None,
            cpu_duration,
        };

        if let Some(timing) = self.timing.take() {
            unsafe {
                self.encoder.sample(&timing.sample_buffer, timing.end_index, Bool::YES);
            }
            record.timing = GpuTiming::Counters {
                sample_buffer: timing.sample_buffer,
                start_index: timing.start_index,
                end_index: timing.end_index,
            };
        }

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
            self.process_completion_inner();
        });
    }

    fn process_completion_inner(&self) {
        let records = self.take_records();
        if records.is_empty() {
            registry().lock().expect("registry mutex poisoned").remove(&self.key);
            return;
        }

        for record in records {
            let gpu_duration = self.counter.resolve_duration(&record);
            let mut duration = gpu_duration.unwrap_or(record.cpu_duration);
            if duration.is_zero() {
                duration = Duration::from_micros(1);
            }

            if gpu_duration.is_none() {
                tracing::debug!(
                    target: "instrument",
                    op = %record.op_name,
                    backend = %record.backend,
                    "falling back to CPU timing for GPU profiler scope"
                );
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
    cpu_duration: Duration,
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
    unsafe fn sample(&self, sample_buffer: &Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>, index: NSUInteger, barrier: Bool) {
        let buffer: &ProtocolObject<dyn MTLCounterSampleBuffer> = sample_buffer.as_ref();
        match self {
            EncoderHandle::Compute(encoder) => {
                let _: () = unsafe {
                    msg_send![
                        &**encoder,
                        sampleCountersInBuffer: buffer,
                        atSampleIndex: index,
                        withBarrier: barrier
                    ]
                };
            }
            EncoderHandle::Blit(encoder) => {
                let _: () = unsafe {
                    msg_send![
                        &**encoder,
                        sampleCountersInBuffer: buffer,
                        atSampleIndex: index,
                        withBarrier: barrier
                    ]
                };
            }
        }
    }
}

fn registry() -> &'static Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>> {
    static REGISTRY: OnceLock<Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn buffer_key(command_buffer: &CommandBuffer) -> usize {
    Retained::as_ptr(command_buffer.raw()) as usize
}

impl GpuProfiler {
    pub fn attach(command_buffer: &CommandBuffer) -> Option<Self> {
        let key = buffer_key(command_buffer);
        let raw = command_buffer.raw();
        let device = unsafe { raw.device() };
        let counter = CounterResources::new(Some(device.as_ref()));
        let dispatch = dispatcher::get_default(|dispatch| dispatch.clone());
        let state = Arc::new(GpuProfilerState::new(key, counter, dispatch));

        registry()
            .lock()
            .expect("registry mutex poisoned")
            .insert(key, Arc::downgrade(&state));

        let completion_state = Arc::clone(&state);
        command_buffer.on_completed(move |_| {
            completion_state.process_completion();
        });

        Some(Self { _state: state })
    }

    fn scope_for_encoder(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        state: Arc<GpuProfilerState>,
        encoder: EncoderHandle,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let timing = state.counter.allocate_timing(command_buffer, &encoder);
        if let Some(sample) = timing.as_ref() {
            unsafe {
                encoder.sample(&sample.sample_buffer, sample.start_index, Bool::YES);
            }
        }

        Some(GpuProfilerScope {
            inner: Some(GpuProfilerScopeInner {
                state,
                timing,
                encoder,
                op_name,
                backend,
                cpu_start: Instant::now(),
            }),
        })
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
        Self::scope_for_encoder(
            command_buffer,
            state,
            EncoderHandle::Compute(encoder.clone()),
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
        Self::scope_for_encoder(
            command_buffer,
            state,
            EncoderHandle::Blit(encoder.clone()),
            format!("{op_name}#{sequence}"),
            backend,
        )
    }
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
        let counter_set = self.timestamp_counter_set.as_ref()?;
        match encoder {
            EncoderHandle::Compute(_) if !self.supports_dispatch_sampling && !self.supports_stage_sampling => return None,
            EncoderHandle::Blit(_) if !self.supports_blit_sampling => return None,
            _ => {}
        }

        let descriptor = unsafe { MTLCounterSampleBufferDescriptor::new() };
        unsafe {
            descriptor.setCounterSet(Some(counter_set.as_ref()));
            descriptor.setSampleCount(2);
        }

        let device = unsafe { command_buffer.device() };
        let sample_buffer = unsafe {
            match device.newCounterSampleBufferWithDescriptor_error(&descriptor) {
                Ok(buffer) => buffer,
                Err(_) => return None,
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

        let period = self.timestamp_period?;
        let length = end_index - start_index + 1;
        let data: Retained<NSData> = unsafe { sample_buffer.resolveCounterRange(NSRange::new(*start_index, length))? };
        let bytes = unsafe { data.as_bytes_unchecked() };
        if bytes.len() < core::mem::size_of::<MTLCounterResultTimestamp>() * 2 {
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
        let start = samples.get(start_index)?.timestamp;
        let end = samples.get(end_index)?.timestamp;
        if start == MTLCounterErrorValue || end == MTLCounterErrorValue || end <= start {
            return None;
        }

        let delta = (end - start) as f64 * period;
        if delta <= 0.0 || !delta.is_finite() {
            return None;
        }

        Some(Duration::from_secs_f64(delta))
    }

    fn find_timestamp_counter_set(device: &ProtocolObject<dyn MTLDevice>) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
        let sets = unsafe { device.counterSets() }?;
        let desired: &NSString = unsafe { MTLCommonCounterSetTimestamp };
        let count = sets.count();
        for idx in 0..count {
            let set = sets.objectAtIndex(idx as NSUInteger);
            let name = unsafe { set.name() };
            if unsafe { name.isEqualToString(desired) } {
                return Some(set);
            }
        }
        None
    }

    fn gpu_timestamp_period(device: &ProtocolObject<dyn MTLDevice>) -> Option<f64> {
        let mut info = mach_timebase_info_data_t { numer: 0, denom: 0 };
        if unsafe { mach_timebase_info(&mut info) } != KERN_SUCCESS || info.denom == 0 {
            return None;
        }

        let mut sleep = Duration::from_micros(200);
        for _ in 0..5 {
            let mut cpu_start: u64 = 0;
            let mut gpu_start: u64 = 0;
            unsafe {
                device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_start), NonNull::from(&mut gpu_start));
            }

            std::thread::sleep(sleep);

            let mut cpu_end: u64 = 0;
            let mut gpu_end: u64 = 0;
            unsafe {
                device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu_end), NonNull::from(&mut gpu_end));
            }

            let cpu_delta = cpu_end.saturating_sub(cpu_start);
            let gpu_delta = gpu_end.saturating_sub(gpu_start);
            if cpu_delta != 0 && gpu_delta != 0 {
                let cpu_delta_ns = (cpu_delta as u128 * info.numer as u128) / info.denom as u128;
                let period = cpu_delta_ns as f64 / 1e9 / gpu_delta as f64;
                if period.is_finite() && period > 0.0 {
                    return Some(period);
                }
            }

            sleep = sleep.saturating_mul(2);
        }

        None
    }
}
