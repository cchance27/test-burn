//! GPU command-buffer profiler emitting per-operation metrics.

use crate::event::MetricEvent;
use crate::record_metric;

pub type CommandBufferCompletionHandler = Box<dyn FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) + Send + 'static>;

pub trait ProfiledCommandBuffer {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    fn on_completed(&self, handler: CommandBufferCompletionHandler);
}

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLComputeCommandEncoder};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use tracing::Dispatch;
use tracing::{self, dispatcher};

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
    op_name: String,
    backend: String,
    cpu_start: Instant,
}

impl GpuProfilerScopeInner {
    fn complete(self) {
        let cpu_duration = self.cpu_start.elapsed();
        let record = GpuOpRecord {
            op_name: self.op_name,
            backend: self.backend,
            cpu_duration,
        };

        self.state.push_record(record);
    }
}

struct GpuProfilerState {
    key: usize,
    dispatch: Dispatch,
    records: Mutex<Vec<GpuOpRecord>>,
    sequence: Mutex<u64>,
    command_buffer_timing: Mutex<Option<CommandBufferTiming>>,
    record_command_buffer_timing: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct CommandBufferTiming {
    gpu: Option<Duration>,
    kernel: Option<Duration>,
}

impl CommandBufferTiming {
    fn from_command_buffer(buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Self {
        unsafe {
            Self {
                gpu: host_interval(msg_send![buffer, GPUStartTime], msg_send![buffer, GPUEndTime]),
                kernel: host_interval(msg_send![buffer, kernelStartTime], msg_send![buffer, kernelEndTime]),
            }
        }
    }

    fn best_duration(&self) -> Option<Duration> {
        self.gpu.or(self.kernel)
    }
}

fn host_interval(start: f64, end: f64) -> Option<Duration> {
    if start <= 0.0 || end <= 0.0 {
        return None;
    }
    let delta = end - start;
    if delta <= 0.0 || !delta.is_finite() {
        return None;
    }
    Some(Duration::from_secs_f64(delta))
}

impl GpuProfilerState {
    fn new(key: usize, dispatch: Dispatch, record_command_buffer_timing: bool) -> Self {
        Self {
            key,
            dispatch,
            records: Mutex::new(Vec::new()),
            sequence: Mutex::new(0),
            command_buffer_timing: Mutex::new(None),
            record_command_buffer_timing,
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

        let command_buffer_duration = self.command_buffer_runtime();
        for record in records {
            let mut duration = if let Some(fallback) = command_buffer_duration {
                fallback
            } else {
                record.cpu_duration
            };
            if duration.is_zero() {
                duration = Duration::from_micros(1);
            }

            if command_buffer_duration.is_some() {
                tracing::debug!(
                    target: "instrument",
                    op = %record.op_name,
                    backend = %record.backend,
                    "falling back to command buffer timing for GPU profiler scope"
                );
            } else {
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

    fn command_buffer_runtime(&self) -> Option<Duration> {
        if !self.record_command_buffer_timing {
            return None;
        }

        self.command_buffer_timing
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(CommandBufferTiming::best_duration))
    }

    fn record_command_buffer_timing(&self, timing: CommandBufferTiming) {
        if !self.record_command_buffer_timing {
            return;
        }

        if let Ok(mut slot) = self.command_buffer_timing.lock() {
            *slot = Some(timing);
        }
    }
}

#[derive(Clone)]
struct GpuOpRecord {
    op_name: String,
    backend: String,
    cpu_duration: Duration,
}

// SAFETY: `GpuOpRecord` only contains owned data and `Duration`, which are safe to
// share across threads as documented above.
unsafe impl Send for GpuOpRecord {}

// SAFETY: See `Send` rationale.
unsafe impl Sync for GpuOpRecord {}

fn registry() -> &'static Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>> {
    static REGISTRY: OnceLock<Mutex<std::collections::HashMap<usize, Weak<GpuProfilerState>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn buffer_key<C: ProfiledCommandBuffer + ?Sized>(command_buffer: &C) -> usize {
    Retained::as_ptr(command_buffer.raw()) as usize
}

impl GpuProfiler {
    pub fn attach<C: ProfiledCommandBuffer + ?Sized>(command_buffer: &C, record_command_buffer_timing: bool) -> Option<Self> {
        let key = buffer_key(command_buffer);
        let dispatch = dispatcher::get_default(|dispatch| dispatch.clone());
        let state = Arc::new(GpuProfilerState::new(key, dispatch, record_command_buffer_timing));

        registry()
            .lock()
            .expect("registry mutex poisoned")
            .insert(key, Arc::downgrade(&state));

        let completion_state = Arc::clone(&state);
        command_buffer.on_completed(Box::new(move |completed_buffer| {
            completion_state.record_command_buffer_timing(CommandBufferTiming::from_command_buffer(completed_buffer));
            completion_state.process_completion();
        }));

        Some(Self { _state: state })
    }

    fn scope_for_encoder(state: Arc<GpuProfilerState>, op_name: String, backend: String) -> Option<GpuProfilerScope> {
        Some(GpuProfilerScope {
            inner: Some(GpuProfilerScopeInner {
                state,
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
        _encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_encoder(state, format!("{op_name}#{sequence}"), backend)
    }

    pub fn profile_blit(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _encoder: &Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_encoder(state, format!("{op_name}#{sequence}"), backend)
    }

    pub fn profile_command_buffer(
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        op_name: String,
        backend: String,
    ) -> Option<GpuProfilerScope> {
        let state = Self::from_command_buffer(command_buffer)?;
        let sequence = state.next_sequence();
        Self::scope_for_encoder(state, format!("{op_name}#{sequence}"), backend)
    }
}
