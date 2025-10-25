use std::sync::{Arc, Mutex};

use metallic_instrumentation::{MetricEvent, record_metric_async};
use objc2::rc::Retained;
use rustc_hash::FxHashMap;

use super::{CommandBuffer, Tensor};
use crate::TensorElement;

/// Cache key for identifying a tensor uniquely
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorCacheKey {
    buffer_ptr: usize,
    offset: usize,
    tensor_id: u64, // Additional identifier to distinguish tensor instances with same buffer
}

impl TensorCacheKey {
    pub fn from_tensor<T: TensorElement>(tensor: &Tensor<T>) -> Self {
        let buffer_ptr = Retained::as_ptr(&tensor.buf) as *const _ as usize;
        let tensor_id = tensor as *const Tensor<T> as u64; // Use pointer to tensor as unique ID
        Self {
            buffer_ptr,
            offset: tensor.offset,
            tensor_id,
        }
    }
}

/// Information about a tensor's preparation state
#[derive(Clone, Debug)]
pub struct TensorPreparationState {
    cmd_buffer_ptr: usize,
}

impl TensorPreparationState {
    pub fn new(cmd_buffer: &CommandBuffer) -> Self {
        let cmd_buffer_ptr = cmd_buffer as *const CommandBuffer as usize;
        Self { cmd_buffer_ptr }
    }
}

/// Performance metrics tracked for tensor preparation
#[derive(Clone, Debug, Default)]
pub struct TensorPreparationMetrics {
    /// Number of times tensor preparation was skipped due to cache hit
    pub cache_hits: u64,
    /// Number of times tensor preparation was performed (cache miss)
    pub cache_misses: u64,
    /// Total time spent performing actual preparation work (microseconds)
    pub total_preparation_time_us: u64,
    /// Total time spent when cache hits short-circuit (microseconds)
    pub total_hit_time_us: u64,
    /// Time saved by cache hits (microseconds) - estimated from running averages
    pub estimated_time_saved_us: u64,
    avg_miss_time_us: f64,
    avg_hit_time_us: f64,
    miss_samples: u64,
    hit_samples: u64,
}

/// Cache for tensor preparation states to avoid redundant preparation operations
///
/// This cache helps optimize tensor preparation in hotpath operations by avoiding
/// redundant preparation work for tensors that don't change between calls.
#[derive(Default)]
pub struct TensorPreparationCache<T: TensorElement> {
    /// Cache of tensor preparation states
    tensor_states: Arc<Mutex<FxHashMap<TensorCacheKey, TensorPreparationState>>>,
    /// Performance metrics tracking
    metrics: Arc<Mutex<TensorPreparationMetrics>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TensorElement> TensorPreparationCache<T> {
    pub fn new() -> Self {
        Self {
            tensor_states: Arc::new(Mutex::new(FxHashMap::default())),
            metrics: Arc::new(Mutex::new(TensorPreparationMetrics::default())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if a tensor's preparation state is still valid for the current command buffer
    ///
    /// This method determines whether the tensor has already been prepared for the
    /// current command buffer and doesn't require preparation again.
    pub fn is_prepared(&self, tensor: &Tensor<T>, current_cmd_buffer: &CommandBuffer) -> bool {
        let key = TensorCacheKey::from_tensor(tensor);
        let current_cmd_buffer_ptr = current_cmd_buffer as *const CommandBuffer as usize;

        let lock = self.tensor_states.lock().expect("Tensor preparation cache mutex poisoned");
        lock.get(&key)
            .map(|state| state.cmd_buffer_ptr == current_cmd_buffer_ptr)
            .unwrap_or(false)
    }

    /// Mark a tensor as prepared for the current command buffer
    pub fn mark_prepared(&self, tensor: &Tensor<T>, cmd_buffer: &CommandBuffer) {
        let key = TensorCacheKey::from_tensor(tensor);
        let state = TensorPreparationState::new(cmd_buffer);

        let mut lock = self.tensor_states.lock().expect("Tensor preparation cache mutex poisoned");
        lock.insert(key, state);
    }

    /// Clear the preparation state for a tensor (e.g., when it's modified)
    pub fn mark_dirty<U: TensorElement>(&self, tensor: &Tensor<U>) {
        let key = TensorCacheKey::from_tensor(tensor);
        let mut lock = self.tensor_states.lock().expect("Tensor preparation cache mutex poisoned");
        lock.remove(&key);
    }

    /// Clear all cached preparation states
    pub fn clear(&self) {
        let mut lock = self.tensor_states.lock().expect("Tensor preparation cache mutex poisoned");
        lock.clear();

        // Also reset metrics
        let mut metrics = self.metrics.lock().expect("Metrics mutex poisoned");
        *metrics = TensorPreparationMetrics::default();
    }

    /// Validate that cached tensors are still valid (e.g., after command buffer completion)
    pub fn validate_states(&self, cmd_buffer: &CommandBuffer) {
        let cmd_buffer_ptr = cmd_buffer as *const CommandBuffer as usize;
        let mut lock = self.tensor_states.lock().expect("Tensor preparation cache mutex poisoned");

        // Remove states for the completed command buffer to force re-preparation
        // This ensures tensors prepared for a completed command buffer are prepared again
        // for new command buffers
        let keys_to_remove: Vec<_> = lock
            .iter()
            .filter(|(_, state)| state.cmd_buffer_ptr == cmd_buffer_ptr)
            .map(|(k, _)| *k)
            .collect();

        for key in keys_to_remove {
            lock.remove(&key);
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> TensorPreparationMetrics {
        self.metrics.lock().expect("Metrics mutex poisoned").clone()
    }

    /// Record the duration of a cache hit (microseconds)
    pub fn record_cache_hit(&self, duration_us: u64) {
        let mut metrics = self.metrics.lock().expect("Metrics mutex poisoned");
        metrics.cache_hits = metrics.cache_hits.saturating_add(1);
        metrics.total_hit_time_us = metrics.total_hit_time_us.saturating_add(duration_us);
        metrics.hit_samples = metrics.hit_samples.saturating_add(1);
        metrics.avg_hit_time_us = update_running_average(metrics.avg_hit_time_us, metrics.hit_samples, duration_us);
        metrics.estimated_time_saved_us = estimate_time_saved(&metrics);

        let should_emit = should_emit_metrics(&metrics, duration_us);
        let event = if should_emit { Some(build_metric_event(&metrics)) } else { None };
        drop(metrics);

        if let Some(event) = event {
            record_metric_async!(event);
        }
    }

    /// Record the duration of a cache miss (microseconds)
    pub fn record_cache_miss(&self, duration_us: u64) {
        let mut metrics = self.metrics.lock().expect("Metrics mutex poisoned");
        metrics.cache_misses = metrics.cache_misses.saturating_add(1);
        metrics.total_preparation_time_us = metrics.total_preparation_time_us.saturating_add(duration_us);
        metrics.miss_samples = metrics.miss_samples.saturating_add(1);
        metrics.avg_miss_time_us = update_running_average(metrics.avg_miss_time_us, metrics.miss_samples, duration_us);
        metrics.estimated_time_saved_us = estimate_time_saved(&metrics);

        let should_emit = should_emit_metrics(&metrics, duration_us);
        let event = if should_emit { Some(build_metric_event(&metrics)) } else { None };
        drop(metrics);

        if let Some(event) = event {
            record_metric_async!(event);
        }
    }

    /// Periodically report current metrics to instrumentation system
    pub fn report_metrics(&self) {
        let metrics = self.metrics.lock().expect("Metrics mutex poisoned");

        // Clone metrics to avoid holding the lock during async recording
        let current_metrics = metrics.clone();
        std::mem::drop(metrics);

        record_metric_async!(build_metric_event(&current_metrics));
    }
}

impl<T: TensorElement> Clone for TensorPreparationCache<T> {
    fn clone(&self) -> Self {
        Self {
            tensor_states: self.tensor_states.clone(),
            metrics: self.metrics.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

fn update_running_average(current_avg: f64, samples: u64, sample_us: u64) -> f64 {
    if samples == 0 {
        return current_avg;
    }

    let sample = sample_us as f64;
    if samples == 1 {
        sample
    } else {
        current_avg + (sample - current_avg) / samples as f64
    }
}

fn estimate_time_saved(metrics: &TensorPreparationMetrics) -> u64 {
    if metrics.hit_samples == 0 || metrics.miss_samples == 0 {
        return 0;
    }

    let per_hit = (metrics.avg_miss_time_us - metrics.avg_hit_time_us).max(0.0);
    (per_hit * metrics.cache_hits as f64).round() as u64
}

fn should_emit_metrics(metrics: &TensorPreparationMetrics, duration_us: u64) -> bool {
    let total_ops = metrics.cache_hits + metrics.cache_misses;
    total_ops.is_multiple_of(100) || duration_us > 1_000
}

fn build_metric_event(metrics: &TensorPreparationMetrics) -> MetricEvent {
    let total_ops = metrics.cache_hits + metrics.cache_misses;
    let hit_rate = if total_ops == 0 {
        0.0
    } else {
        (metrics.cache_hits as f64 / total_ops as f64) * 100.0
    };

    MetricEvent::TensorPreparationStats {
        cache_hits: metrics.cache_hits,
        cache_misses: metrics.cache_misses,
        total_preparation_time_us: metrics.total_preparation_time_us,
        estimated_time_saved_us: metrics.estimated_time_saved_us,
        hit_rate,
    }
}
