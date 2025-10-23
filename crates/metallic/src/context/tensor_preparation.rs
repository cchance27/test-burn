use super::main::Context;
use crate::{
    MetalError, tensor::{Tensor, TensorElement}
};

impl<T: TensorElement> Context<T> {
    fn prepare_tensor_for_active_cmd(&mut self, tensor: &Tensor<T>) -> Result<(), MetalError> {
        // Record start time for metrics
        let start_time = std::time::Instant::now();

        // Always flush host writes first so GPU sees the most recent data
        tensor.flush_host_writes()?;

        // First, check if tensor is already prepared for current command buffer
        if let Some(active_cmd_buffer) = self.active_cmd_buffer.as_ref()
            && self.tensor_preparation_cache.is_prepared(tensor, active_cmd_buffer)
        {
            // Tensor is already prepared; record cache hit metrics and return
            let elapsed_us = start_time.elapsed().as_micros().max(1) as u64;
            self.tensor_preparation_cache.record_cache_hit(elapsed_us);
            return Ok(());
        }

        // Run the original preparation logic to maintain correctness
        let maybe_dep = tensor.defining_cmd_buffer.borrow().clone();
        if let Some(dep) = maybe_dep {
            if let Some(active_cmd_buffer) = self.active_cmd_buffer.as_ref() {
                if self.active_cmd_buffer.as_ref().map(|active| dep.ptr_eq(active)).unwrap_or(false) {
                    // If tensor is already pending on our active command buffer, we're done
                    // Still mark in cache for tracking purposes
                    self.tensor_preparation_cache.mark_prepared(tensor, active_cmd_buffer);

                    // Record timing for this preparation attempt
                    let elapsed_us = start_time.elapsed().as_micros().max(1) as u64;
                    self.tensor_preparation_cache.record_cache_miss(elapsed_us);
                    return Ok(());
                }

                if dep.is_completed() {
                    tensor.defining_cmd_buffer.borrow_mut().take();
                    // Mark tensor as prepared for the current command buffer
                    self.tensor_preparation_cache.mark_prepared(tensor, active_cmd_buffer);

                    // Record timing for this preparation attempt
                    let elapsed_us = start_time.elapsed().as_micros().max(1) as u64;
                    self.tensor_preparation_cache.record_cache_miss(elapsed_us);
                    return Ok(());
                }

                // Attribute dependency wait to current logical GPU scope so it doesn't show up as Other
                let wait_start = std::time::Instant::now();
                dep.commit();
                dep.wait();
                let waited = wait_start.elapsed();
                if !waited.is_zero() {
                    if let Some(label) = self.current_gpu_scope_label() {
                        let path = label.op_name;
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: format!("{}/dep_wait", path),
                            backend: label.backend,
                            duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                        });
                    } else {
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: "Generation Loop/dep_wait".to_string(),
                            backend: super::utils::GPU_PROFILER_BACKEND.to_string(),
                            duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                        });
                    }
                }

                tensor.defining_cmd_buffer.borrow_mut().take();
            } else {
                // When no active command buffer exists, just run original preparation logic
                if dep.is_completed() {
                    tensor.defining_cmd_buffer.borrow_mut().take();
                    // Record timing for this preparation attempt
                    let elapsed_us = start_time.elapsed().as_micros().max(1) as u64;
                    self.tensor_preparation_cache.record_cache_miss(elapsed_us);
                    return Ok(());
                }

                // Attribute dependency wait to current logical GPU scope so it doesn't show up as Other
                let wait_start = std::time::Instant::now();
                dep.commit();
                dep.wait();
                let waited = wait_start.elapsed();
                if !waited.is_zero() {
                    if let Some(label) = self.current_gpu_scope_label() {
                        let path = label.op_name;
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: format!("{}/dep_wait", path),
                            backend: label.backend,
                            duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                        });
                    } else {
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: "Generation Loop/dep_wait".to_string(),
                            backend: super::utils::GPU_PROFILER_BACKEND.to_string(),
                            duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                        });
                    }
                }

                tensor.defining_cmd_buffer.borrow_mut().take();
            }
        }

        // Mark tensor as prepared if we have an active command buffer
        if let Some(active_cmd_buffer) = self.active_cmd_buffer.as_ref() {
            self.tensor_preparation_cache.mark_prepared(tensor, active_cmd_buffer);
        }

        // Record timing for this preparation attempt
        let elapsed_us = start_time.elapsed().as_micros().max(1) as u64;
        self.tensor_preparation_cache.record_cache_miss(elapsed_us);

        Ok(())
    }

    #[inline]
    pub(crate) fn prepare_tensors_for_active_cmd(&mut self, tensors: &[&Tensor<T>]) -> Result<(), MetalError> {
        for tensor in tensors {
            self.prepare_tensor_for_active_cmd(tensor)?;
        }
        Ok(())
    }
}
