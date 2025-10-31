use std::{
    cell::RefCell, collections::VecDeque, rc::{Rc, Weak}, sync::{Arc, Mutex}, time::Instant
};

use metallic_instrumentation::MetricEvent;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandQueue;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

use super::utils::{GPU_PROFILER_BACKEND, GpuProfilerLabel};
use crate::{error::MetalError, operation::CommandBuffer};

const DEFAULT_MAX_INFLIGHT: usize = 3;

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
}

#[derive(Clone)]
pub struct PipelineCompletion {
    pub command_buffer: CommandBuffer,
    pub label: Option<GpuProfilerLabel>,
    pub wait_duration: std::time::Duration,
}

struct PipelineRegistration {
    pipeline: Weak<RefCell<CommandBufferPipelineState>>,
    completion_observer: Option<Arc<dyn Fn(&CommandBuffer) + Send + Sync + 'static>>,
}

static PIPELINE_REGISTRY: Lazy<Mutex<FxHashMap<usize, PipelineRegistration>>> = Lazy::new(|| Mutex::new(FxHashMap::default()));

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
        command_buffer.commit();
        state.inflight.push_back(PipelineEntry { command_buffer, label });
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

    pub fn flush_all(&self) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().flush_all_locked()
    }

    pub fn wait_for(&self, target: &CommandBuffer, label: Option<GpuProfilerLabel>) -> Vec<PipelineCompletion> {
        self.inner.borrow_mut().wait_for_locked(target, label)
    }
}

impl CommandBufferPipelineState {
    fn reserve_slot_locked(&mut self) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();
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
        let mut completed = Vec::new();
        while let Some(done) = self.wait_oldest_locked() {
            completed.push(done);
        }
        completed
    }

    fn wait_for_locked(&mut self, target: &CommandBuffer, label: Option<GpuProfilerLabel>) -> Vec<PipelineCompletion> {
        let mut completed = Vec::new();
        target.commit();

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
            self.inflight.push_back(PipelineEntry {
                command_buffer: target.clone(),
                label,
            });
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
            let wait_start = Instant::now();
            entry.command_buffer.wait();
            PipelineCompletion {
                wait_duration: wait_start.elapsed(),
                label: entry.label,
                command_buffer: entry.command_buffer,
            }
        })
    }
}

fn queue_key(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> usize {
    Retained::as_ptr(queue) as *const ProtocolObject<dyn MTLCommandQueue> as usize
}

fn register_pipeline(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>, pipeline: &CommandBufferPipeline) {
    let key = queue_key(queue);
    let mut registry = PIPELINE_REGISTRY.lock().expect("command buffer pipeline registry mutex poisoned");
    let observer = registry.get(&key).and_then(|entry| entry.completion_observer.clone());
    registry.insert(
        key,
        PipelineRegistration {
            pipeline: Rc::downgrade(&pipeline.inner),
            completion_observer: observer,
        },
    );
}

pub fn register_completion_observer(
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    observer: Arc<dyn Fn(&CommandBuffer) + Send + Sync + 'static>,
) {
    let key = queue_key(queue);
    let mut registry = PIPELINE_REGISTRY.lock().expect("command buffer pipeline registry mutex poisoned");
    registry
        .entry(key)
        .and_modify(|entry| entry.completion_observer = Some(observer.clone()))
        .or_insert_with(|| PipelineRegistration {
            pipeline: Weak::new(),
            completion_observer: Some(observer),
        });
}

pub fn lookup_pipeline(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Option<CommandBufferPipeline> {
    let key = queue_key(queue);
    let mut registry = PIPELINE_REGISTRY.lock().expect("command buffer pipeline registry mutex poisoned");
    let Some(entry) = registry.get_mut(&key) else {
        return None;
    };
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
    let observer = {
        let mut registry = PIPELINE_REGISTRY.lock().expect("command buffer pipeline registry mutex poisoned");
        registry.get(&key).and_then(|entry| entry.completion_observer.clone())
    };

    for completion in completions {
        let waited = completion.wait_duration;
        if !waited.is_zero() {
            if let Some(label) = completion.label.as_ref() {
                metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                    op_name: format!("{}/cb_wait", label.op_name),
                    backend: label.backend.clone(),
                    duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                });
            } else {
                metallic_instrumentation::record_metric_async!(MetricEvent::GpuOpCompleted {
                    op_name: "Generation Loop/cb_wait".to_string(),
                    backend: GPU_PROFILER_BACKEND.to_string(),
                    duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                });
            }
        }
    }

    if let Some(observer) = observer {
        for completion in completions {
            observer(&completion.command_buffer);
        }
    }
}
