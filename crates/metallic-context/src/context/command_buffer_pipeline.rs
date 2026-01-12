use std::{
    cell::RefCell, collections::VecDeque, rc::{Rc, Weak}, sync::{
        Arc, Mutex, OnceLock, atomic::{AtomicBool, Ordering}
    }, time::Instant
};

use metallic_instrumentation::MetricEvent;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use rustc_hash::FxHashMap;

use super::utils::{GPU_PROFILER_BACKEND, GpuProfilerLabel};
use crate::{error::MetalError, operation::CommandBuffer};

const DEFAULT_MAX_INFLIGHT: usize = 3;
const METALLIC_RECORD_CB_GPU_TIMING_ENV: &str = "METALLIC_RECORD_CB_GPU_TIMING";

#[inline]
fn record_cb_gpu_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(METALLIC_RECORD_CB_GPU_TIMING_ENV)
            .ok()
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

thread_local! {
    static PIPELINE_REGISTRY: RefCell<FxHashMap<usize, PipelineRegistration>> =
        RefCell::new(FxHashMap::default());
}

#[derive(Clone)]
pub struct CommandBufferPipeline {
    inner: Rc<RefCell<CommandBufferPipelineState>>,
}

struct CommandBufferPipelineState {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    inflight: VecDeque<PipelineEntry>,
    max_inflight: usize,
}

#[derive(Clone)]
struct PipelineEntry {
    command_buffer: CommandBuffer,
    label: Option<GpuProfilerLabel>,
    completion_flag: Arc<AtomicBool>,
    completion_time: Arc<Mutex<Option<Instant>>>,
    commit_instant: Instant,
}

#[derive(Clone)]
pub struct PipelineCompletion {
    pub command_buffer: CommandBuffer,
    pub label: Option<GpuProfilerLabel>,
    pub wait_duration: std::time::Duration,
}

type CompletionObserver = Arc<dyn Fn(&CommandBuffer) + 'static>;

struct PipelineRegistration {
    pipeline: Weak<RefCell<CommandBufferPipelineState>>,
    completion_observer: Option<CompletionObserver>,
}

impl CommandBufferPipeline {
    pub fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        let inner = Rc::new(RefCell::new(CommandBufferPipelineState {
            queue: queue.clone(),
            inflight: VecDeque::with_capacity(DEFAULT_MAX_INFLIGHT),
            max_inflight: DEFAULT_MAX_INFLIGHT,
        }));
        let pipeline = Self { inner };
        register_pipeline(&queue, &pipeline);
        pipeline
    }

    fn from_inner(inner: Rc<RefCell<CommandBufferPipelineState>>) -> Self {
        Self { inner }
    }

    pub fn submit(&self, command_buffer: CommandBuffer, label: Option<GpuProfilerLabel>) {
        let mut state = self.inner.borrow_mut();
        let completion_flag = Arc::new(AtomicBool::new(false));
        let completion_time = Arc::new(Mutex::new(None));
        let completion_flag_clone = completion_flag.clone();
        let completion_time_clone = completion_time.clone();
        command_buffer.on_completed(Box::new({
            move |_: &ProtocolObject<dyn MTLCommandBuffer>| {
                completion_flag_clone.store(true, Ordering::Release);
                if let Ok(mut slot) = completion_time_clone.lock() {
                    *slot = Some(Instant::now());
                }
            }
        }));
        let commit_instant = Instant::now();
        command_buffer.commit();
        state.inflight.push_back(PipelineEntry {
            command_buffer,
            label,
            completion_flag,
            completion_time,
            commit_instant,
        });
    }

    pub fn acquire(&self) -> Result<(CommandBuffer, Vec<PipelineCompletion>), MetalError> {
        let mut state = self.inner.borrow_mut();
        let completed = state.reserve_slot_locked();
        let command_buffer = CommandBuffer::new(&state.queue)?;
        Ok((command_buffer, completed))
    }

    pub fn reserve_slot(&self) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().reserve_slot_locked()
    }

    pub fn collect_completed(&self) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().collect_completed_locked()
    }

    pub fn flush_all(&self) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().flush_all_locked()
    }

    pub fn wait_for(&self, target: &CommandBuffer, label: Option<GpuProfilerLabel>) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().wait_for_locked(target, label)
    }
}

impl CommandBufferPipelineState {
    fn reserve_slot_locked(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = self.collect_completed_locked();
        while self.inflight.len() >= self.max_inflight {
            if let Some(done) = self.wait_oldest_locked() {
                completed.push(done);
            } else {
                break;
            }
        }
        completed
    }

    fn flush_all_locked(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = self.collect_completed_locked();
        while let Some(done) = self.wait_oldest_locked() {
            completed.push(done);
        }
        completed
    }

    fn collect_completed_locked(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();
        while let Some(front) = self.inflight.front() {
            if !front.completion_flag.load(Ordering::Acquire) {
                break;
            }
            let entry = self.inflight.pop_front().expect("front entry must exist");
            entry.command_buffer.wait();
            let wait_duration = entry
                .completion_time
                .lock()
                .ok()
                .and_then(|mut slot| slot.take())
                .map(|finished| finished.saturating_duration_since(entry.commit_instant))
                .unwrap_or_else(|| entry.commit_instant.elapsed());
            completed.push(PipelineCompletion {
                command_buffer: entry.command_buffer,
                label: entry.label,
                wait_duration,
            });
        }
        completed
    }

    fn wait_for_locked(&mut self, target: &CommandBuffer, label: Option<GpuProfilerLabel>) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();

        let mut tracked = false;
        for entry in self.inflight.iter_mut() {
            if entry.command_buffer.ptr_eq(target) {
                tracked = true;
                if entry.label.is_none() {
                    entry.label = label.clone();
                }
                break;
            }
        }

        if !tracked {
            while self.inflight.len() >= self.max_inflight {
                if let Some(done) = self.wait_oldest_locked() {
                    completed.push(done);
                }
            }
            if target.is_committed() {
                let wait_start = Instant::now();
                target.wait();
                completed.push(PipelineCompletion {
                    command_buffer: target.clone(),
                    label,
                    wait_duration: wait_start.elapsed(),
                });
                return completed;
            }
            let completion_flag = Arc::new(AtomicBool::new(false));
            let completion_time = Arc::new(Mutex::new(None));
            let completion_flag_clone = completion_flag.clone();
            let completion_time_clone = completion_time.clone();
            target.on_completed(Box::new({
                move |_: &ProtocolObject<dyn MTLCommandBuffer>| {
                    completion_flag_clone.store(true, Ordering::Release);
                    if let Ok(mut slot) = completion_time_clone.lock() {
                        *slot = Some(Instant::now());
                    }
                }
            }));
            let commit_instant = Instant::now();
            target.commit();
            self.inflight.push_back(PipelineEntry {
                command_buffer: target.clone(),
                label,
                completion_flag,
                completion_time,
                commit_instant,
            });
        } else if !target.is_committed() {
            target.commit();
        }

        while let Some(front) = self.inflight.front() {
            if front.command_buffer.ptr_eq(target) {
                if let Some(done) = self.wait_oldest_locked() {
                    completed.push(done);
                }
                break;
            } else if let Some(done) = self.wait_oldest_locked() {
                completed.push(done);
            } else {
                break;
            }
        }

        completed
    }

    fn wait_oldest_locked(&mut self) -> Option<PipelineCompletion> {
        self.inflight.pop_front().map(|entry| {
            entry.command_buffer.wait();
            let wait_duration = entry
                .completion_time
                .lock()
                .ok()
                .and_then(|mut slot| slot.take())
                .map(|finished| finished.saturating_duration_since(entry.commit_instant))
                .unwrap_or_else(|| entry.commit_instant.elapsed());
            PipelineCompletion {
                wait_duration,
                label: entry.label,
                command_buffer: entry.command_buffer,
            }
        })
    }
}

fn queue_key(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> usize {
    Retained::as_ptr(queue) as usize
}

fn register_pipeline(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>, pipeline: &CommandBufferPipeline) {
    let key = queue_key(queue);
    PIPELINE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let observer = registry.get(&key).and_then(|entry| entry.completion_observer.clone());
        registry.insert(
            key,
            PipelineRegistration {
                pipeline: Rc::downgrade(&pipeline.inner),
                completion_observer: observer,
            },
        );
    });
}

pub fn register_completion_observer(
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    observer: Arc<dyn Fn(&CommandBuffer) + 'static>,
) {
    let key = queue_key(queue);
    PIPELINE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry
            .entry(key)
            .and_modify(|entry| entry.completion_observer = Some(observer.clone()))
            .or_insert_with(|| PipelineRegistration {
                pipeline: Weak::new(),
                completion_observer: Some(observer),
            });
    });
}

pub fn lookup_pipeline(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Option<CommandBufferPipeline> {
    let key = queue_key(queue);
    PIPELINE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let entry = registry.get_mut(&key)?;
        if let Some(inner) = entry.pipeline.upgrade() {
            Some(CommandBufferPipeline::from_inner(inner))
        } else {
            if entry.completion_observer.is_none() {
                registry.remove(&key);
            } else {
                entry.pipeline = Weak::new();
            }
            None
        }
    })
}

pub fn wait_with_pipeline(
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    command_buffer: &CommandBuffer,
    label: Option<GpuProfilerLabel>,
) -> Vec<PipelineCompletion> {
    if let Some(pipeline) = lookup_pipeline(queue) {
        pipeline.wait_for(command_buffer, label)
    } else {
        let wait_start = Instant::now();
        command_buffer.wait();
        vec![PipelineCompletion {
            wait_duration: wait_start.elapsed(),
            label,
            command_buffer: command_buffer.clone(),
        }]
    }
}

pub fn dispatch_completions(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>, completions: &[PipelineCompletion]) {
    if completions.is_empty() {
        return;
    }

    let key = queue_key(queue);
    let observer = PIPELINE_REGISTRY.with(|registry| registry.borrow().get(&key).and_then(|entry| entry.completion_observer.clone()));
    let record_cb_gpu = record_cb_gpu_timing_enabled();

    for completion in completions {
        let waited = completion.wait_duration;
        if !waited.is_zero() {
            if let Some(label) = completion.label.as_ref() {
                metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                    op_name: format!("{}/cb_wait", label.op_name),
                    backend: label.backend.clone(),
                    duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    data: None,
                });
            } else {
                metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                    op_name: "Unscoped/cb_wait".to_string(),
                    backend: GPU_PROFILER_BACKEND.to_string(),
                    duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    data: None,
                });
            }
        }

        if record_cb_gpu {
            let cb = completion.command_buffer.raw();
            let cb_us = metallic_instrumentation::gpu_profiler::command_buffer_best_duration_us(cb.as_ref());
            if let Some(duration_us) = cb_us {
                if let Some(label) = completion.label.as_ref() {
                    metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/cb_gpu", label.op_name),
                        backend: label.backend.clone(),
                        duration_us,
                        data: None,
                    });
                } else {
                    metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                        op_name: "Unscoped/cb_gpu".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us,
                        data: None,
                    });
                }
            }
        }
    }

    if let Some(observer) = observer {
        for completion in completions {
            observer(&completion.command_buffer);
        }
    }
}
