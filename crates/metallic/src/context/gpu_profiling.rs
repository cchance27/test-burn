use metallic_instrumentation::MetricEvent;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

use super::{
    command_buffer_pipeline::PipelineCompletion, main::Context, utils::{GPU_PROFILER_BACKEND, GpuProfilerLabel}
};
use crate::tensor::TensorElement;

struct GpuScopeGuard<T: TensorElement> {
    ctx: *mut Context<T>,
}

impl<T: TensorElement> Drop for GpuScopeGuard<T> {
    fn drop(&mut self) {
        // SAFETY: guard is created from an exclusive reference to the context.
        let ctx = unsafe { &mut *self.ctx };
        ctx.gpu_scope_stack.pop().expect("GPU scope stack underflow on guard drop");
    }
}

impl<T: TensorElement> Context<T> {
    #[inline]
    pub fn set_pending_gpu_scope<S: Into<String>>(&mut self, op_name: S) {
        self.pending_gpu_scope = Some(GpuProfilerLabel::new(op_name.into(), GPU_PROFILER_BACKEND.to_string()));
    }

    #[inline]
    pub fn override_pending_gpu_backend<S: Into<String>>(&mut self, backend: S) {
        if let Some(scope) = self.pending_gpu_scope.as_mut() {
            scope.backend = backend.into();
        }
    }

    #[inline]
    pub fn with_gpu_scope<R, F>(&mut self, op_name: impl Into<String>, f: F) -> R
    where
        F: FnOnce(&mut Context<T>) -> R,
    {
        let label = GpuProfilerLabel::new(op_name.into(), GPU_PROFILER_BACKEND.to_string());
        self.gpu_scope_stack.push(label);
        let guard = GpuScopeGuard {
            ctx: self as *mut Context<T>,
        };
        let result = f(self);
        drop(guard);
        result
    }

    #[inline]
    pub fn clear_pending_gpu_scope(&mut self) {
        self.pending_gpu_scope = None;
    }

    /// Build the current hierarchical GPU scope label without consuming the pending scope.
    /// This mirrors `take_gpu_scope` but returns a label even when called from pre-encode phases
    /// such as tensor preparation, so we can attribute CPU-side work (e.g. dependency waits)
    /// to the correct logical op path.
    #[inline]
    pub(crate) fn current_gpu_scope_label(&self) -> Option<GpuProfilerLabel> {
        let mut path_segments = Vec::new();
        for scope in &self.gpu_scope_stack {
            path_segments.push(scope.op_name.clone());
        }
        if let Some(pending_scope) = &self.pending_gpu_scope {
            path_segments.push(pending_scope.op_name.clone());
        }
        if path_segments.is_empty() {
            return None;
        }
        let op_name = path_segments.join("/");
        let backend = self
            .gpu_scope_stack
            .last()
            .map(|s| s.backend.clone())
            .unwrap_or_else(|| GPU_PROFILER_BACKEND.to_string());
        Some(GpuProfilerLabel::new(op_name, backend))
    }

    #[inline]
    pub(crate) fn take_gpu_scope(&mut self) -> Option<GpuProfilerLabel> {
        let mut path_segments = Vec::new();
        for scope in &self.gpu_scope_stack {
            path_segments.push(scope.op_name.clone());
        }

        if let Some(pending_scope) = self.pending_gpu_scope.take() {
            path_segments.push(pending_scope.op_name);
        }

        if path_segments.is_empty() {
            return None;
        }

        let op_name = path_segments.join("/");
        let backend = self
            .gpu_scope_stack
            .last()
            .map(|s| s.backend.clone())
            .unwrap_or_else(|| GPU_PROFILER_BACKEND.to_string());

        Some(GpuProfilerLabel::new(op_name, backend))
    }

    /// Synchronize pending GPU work, committing and waiting on the active command buffer.
    /// Falls back to the legacy submit/wait path if no active buffer exists.
    pub fn synchronize(&mut self) {
        if let Some(cmd_buf) = self.active_cmd_buffer.take() {
            let label = self.current_gpu_scope_label();
            self.command_buffer_pipeline.submit(cmd_buf, label);
        }

        let completed = self.command_buffer_pipeline.flush_all();
        self.process_pipeline_completions(completed);

        if let Some(cb) = self.command_queue.commandBuffer() {
            let wait_start = std::time::Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            let waited = wait_start.elapsed();
            if !waited.is_zero() {
                if let Some(label) = self.current_gpu_scope_label() {
                    let path = label.op_name;
                    metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/cb_wait", path),
                        backend: label.backend,
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                } else {
                    metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: "Generation Loop/cb_wait".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                }
            }
        }
    }

    pub(crate) fn finalize_active_command_buffer_if_latency(&mut self) {
        if !crate::profiling_state::get_profiling_state() {
            return;
        }

        if let Some(cmd_buf) = self.active_cmd_buffer.take() {
            let label = self.current_gpu_scope_label();
            self.command_buffer_pipeline.submit(cmd_buf, label);
        }

        let completed = self.command_buffer_pipeline.flush_all();
        self.process_pipeline_completions(completed);
    }

    pub(crate) fn process_pipeline_completions(&mut self, completions: Vec<PipelineCompletion>) {
        command_buffer_pipeline::dispatch_completions(&self.command_queue, &completions);
    }
}
