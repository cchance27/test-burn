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

use tracing::{self, dispatcher};
use tracing::{Dispatch, trace};

#[derive(Clone)]
pub struct GpuProfiler {
    // Keep the profiler state alive for the lifetime of this handle so scopes
    // may continue to resolve even if the handle is dropped early.
    state: Arc<GpuProfilerState>,
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
    cpu_commit_instant: Mutex<Option<Instant>>,
    cpu_scope_begin: Mutex<Option<Instant>>,
    use_cpu_scope: std::sync::atomic::AtomicBool,
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
            cpu_commit_instant: Mutex::new(None),
            cpu_scope_begin: Mutex::new(None),
            use_cpu_scope: std::sync::atomic::AtomicBool::new(false),
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

        // Group records by (base op_name without '#sequence', backend) and aggregate CPU durations for weighting
        // This collapses multiple encoder dispatches for the same logical op into a single group
        let mut groups: std::collections::HashMap<(String, String), Vec<Duration>> = std::collections::HashMap::new();
        for rec in records {
            let base_name = rec.op_name.split('#').next().unwrap_or(&rec.op_name).to_string();
            groups.entry((base_name, rec.backend)).or_default().push(rec.cpu_duration);
        }

        let mut command_buffer_duration = self.command_buffer_runtime();
        // If Metal reports an unrealistically small GPU time but we have a CPU commit->complete elapsed,
        // use the CPU elapsed as a fallback approximation of the true kernel time (common when heavy work runs
        // in an internal CB not visible to our profiler).
        if let Some(best) = command_buffer_duration {
            let tiny = best < Duration::from_micros(100);
            if tiny
                && let Ok(mut instant_guard) = self.cpu_commit_instant.lock()
                && let Some(committed_at) = instant_guard.take()
            {
                let elapsed = committed_at.elapsed();
                // If elapsed is meaningfully larger than the tiny reported GPU time, prefer it.
                if elapsed > best.saturating_mul(5) {
                    command_buffer_duration = Some(elapsed);
                }
            }
        }
        trace!(
            "CB_COMPLETE gpu_or_kernel_timing_present={} groups_len={}",
            command_buffer_duration.is_some(),
            groups.len()
        );

        // If we have GPU timing and multiple logical ops, attribute the entire duration to the op
        // with the largest CPU encode time. This approximates the dominant kernel and avoids tiny splits.
        // Optional CPU-side timing override for latency mode: measure from first scope creation to completion
        let cpu_scope_elapsed = if self.record_command_buffer_timing {
            if let Ok(mut begin) = self.cpu_scope_begin.lock() {
                begin.take().map(|start| start.elapsed())
            } else {
                None
            }
        } else {
            None
        };

        let dominant_index: Option<usize> = if command_buffer_duration.is_some() && groups.len() > 1 {
            let mut max_cpu = Duration::from_micros(0);
            let mut idx = 0usize;
            for (i, (_k, cpu_durations)) in groups.iter().enumerate() {
                let cpu_total: Duration = cpu_durations
                    .iter()
                    .copied()
                    .fold(Duration::from_micros(0), |acc, d| acc.saturating_add(d));
                if cpu_total > max_cpu {
                    max_cpu = cpu_total;
                    idx = i;
                }
            }
            Some(idx)
        } else {
            None
        };

        let groups_len = groups.len();
        for (i, ((op_name, backend), cpu_durations)) in groups.into_iter().enumerate() {
            let cpu_total: Duration = cpu_durations
                .iter()
                .copied()
                .fold(Duration::from_micros(0), |acc, d| acc.saturating_add(d));

            // Decide timing source:
            // - If this CB/op was marked as MPS-based, prefer CPU scope elapsed; else
            // - If CPU scope elapsed is available (latency mode), use it; else
            // - If GPU CB timing is available, use it (with dominant heuristic); else
            // - Fallback to aggregated CPU encode duration.
            let use_cpu_scope = self.use_cpu_scope.load(std::sync::atomic::Ordering::Relaxed);
            let duration = if use_cpu_scope {
                if let Some(cpu_elapsed) = cpu_scope_elapsed {
                    cpu_elapsed
                } else if let Some(cb_total) = command_buffer_duration {
                    cb_total
                } else {
                    cpu_total
                }
            } else if let Some(cpu_elapsed) = cpu_scope_elapsed {
                cpu_elapsed
            } else if let Some(cb_total) = command_buffer_duration {
                trace!(
                    "CB_OP op={} backend={} cb_total_us={} dominant={}",
                    op_name,
                    backend,
                    (cb_total.as_secs_f64() * 1e6) as u64,
                    Some(i) == dominant_index
                );
                if groups_len == 1 || Some(i) == dominant_index {
                    cb_total
                } else {
                    // Non-dominant ops get a minimal placeholder to show activity
                    Duration::from_micros(1)
                }
            } else {
                // No GPU timing, fallback to aggregated CPU duration
                cpu_total
            };

            let mut duration = duration;
            if duration.is_zero() {
                duration = Duration::from_micros(1);
            }

            let duration_us = (duration.as_secs_f64() * 1e6).max(1.0).round() as u64;
            record_metric!(MetricEvent::GpuOpCompleted {
                op_name,
                backend,
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

    fn note_cpu_commit(&self) {
        if let Ok(mut slot) = self.cpu_commit_instant.lock() {
            *slot = Some(Instant::now());
        }
    }

    fn record_cpu_fallback_if_missing(&self) {
        if !self.record_command_buffer_timing {
            return;
        }
        let mut timing_guard = match self.command_buffer_timing.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let has_best = timing_guard.as_ref().and_then(CommandBufferTiming::best_duration).is_some();
        if has_best {
            return;
        }
        if let Ok(mut instant_guard) = self.cpu_commit_instant.lock()
            && let Some(committed_at) = instant_guard.take()
        {
            let elapsed = committed_at.elapsed();
            *timing_guard = Some(CommandBufferTiming {
                gpu: Some(elapsed),
                kernel: None,
            });
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
    pub fn mark_use_cpu_scope_for_cb(command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        if let Some(state) = Self::from_command_buffer(command_buffer) {
            state.use_cpu_scope.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn note_cpu_commit(&self) {
        self.state.note_cpu_commit();
    }

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
            completion_state.record_cpu_fallback_if_missing();
            completion_state.process_completion();
        }));

        Some(Self { state })
    }

    fn scope_for_encoder(state: Arc<GpuProfilerState>, op_name: String, backend: String) -> Option<GpuProfilerScope> {
        // Mark the CPU scope begin time the first time we create a scope for this CB
        if let Ok(mut begin) = state.cpu_scope_begin.lock()
            && begin.is_none()
        {
            *begin = Some(Instant::now());
        }

        // Mark the CPU scope begin time the first time we create a scope for this CB
        if let Ok(mut begin) = state.cpu_scope_begin.lock()
            && begin.is_none()
        {
            *begin = Some(Instant::now());
        }

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
