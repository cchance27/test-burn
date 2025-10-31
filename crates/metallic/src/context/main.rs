use std::sync::Arc;

use kernels::{KernelBackendOverride, KernelBackendOverrides, KernelBackendRegistry, KernelManager, matmul_mlx::MlxKernelCache};
use metallic_instrumentation::config::AppConfig;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use rustc_hash::FxHashMap;

use super::command_buffer_pipeline::{self, CommandBufferPipeline, PipelineCompletion};
use crate::{
    Tensor, TensorElement, caching::{CacheRegistry, CacheStats, KvCacheEntry, KvCacheState, ResourceCache, TensorPreparationCache}, context::detect_forced_matmul_backend, error::MetalError, kernels::{self, CustomKernelInvocable, DefaultKernelInvocable, MultiTensorOutput}, operation::CommandBuffer, pool::MemoryPool, profiling_state, tensor::Dtype
};

#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepeatKvWorkspaceKind {
    Key,
    Value,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct RepeatKvWorkspaceKey {
    pub layer_idx: usize,
    pub kind: RepeatKvWorkspaceKind,
    pub repeated_heads: usize,
    pub cache_capacity: usize,
    pub head_dim: usize,
    pub dtype: Dtype,
    pub shared: bool,
}

#[derive(Clone)]
pub struct RepeatKvWorkspaceEntry<T: TensorElement> {
    pub tensor: Tensor<T>,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct AliasKey {
    pub buffer_ptr: usize,
    pub offset_bytes: usize,
    pub length_bytes: usize,
}

pub struct Context<T: TensorElement> {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
    pub kernel_manager: KernelManager,
    backend_registry: KernelBackendRegistry,
    // Metrics counters
    pub pooled_bytes_allocated: usize,
    pub pooled_allocations: usize,
    pub pool_resets: usize,
    // RNG seed counter for deterministic random generation
    pub rng_seed_counter: u64,

    pub(crate) cache_registry: CacheRegistry,
    /// Lazily created command buffer used to batch kernel dispatches until synchronization.
    pub(crate) active_cmd_buffer: Option<CommandBuffer>,
    pub(crate) command_buffer_pipeline: CommandBufferPipeline,
    /// Resource cache associated with the active command buffer.
    pub active_resource_cache: Option<ResourceCache>,
    /// Workspace reused across sampling invocations to avoid per-token allocations.
    pub sampler_buffers: SamplerBuffers,
    /// Optional override for the matmul backend chosen by this context.
    pub(crate) forced_matmul_backend: super::utils::MatMulBackendOverride,
    /// Cache of MLX compute pipelines keyed by kernel configuration.
    pub(crate) mlx_kernel_cache: MlxKernelCache,
    pub(crate) sdpa_workspaces: FxHashMap<super::sdpa_workspace::SdpaWorkspaceKey, super::sdpa_workspace::SdpaWorkspaceState>,
    pub(crate) kv_repeat_workspaces: FxHashMap<RepeatKvWorkspaceKey, RepeatKvWorkspaceEntry<T>>,
    pub(crate) pending_gpu_scope: Option<super::utils::GpuProfilerLabel>,
    pub(crate) gpu_scope_stack: Vec<super::utils::GpuProfilerLabel>,
}

impl<T: TensorElement> Context<T> {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device, &command_queue)?;
        let forced_backend = detect_forced_matmul_backend();
        // Initialize the global profiling state based on environment configuration
        profiling_state::initialize_profiling_state_from_env();

        if AppConfig::profiling_forced() {
            profiling_state::set_profiling_state(true);
        }

        let mut cache_registry = CacheRegistry::default();
        {
            let kv_state = KvCacheState::<T>::new(&device, &command_queue)?;
            let _ = cache_registry.slot_mut(|| kv_state);
        }
        {
            let tensor_cache = TensorPreparationCache::<T>::new();
            let _ = cache_registry.slot_mut(|| tensor_cache);
        }

        let pipeline = CommandBufferPipeline::new(command_queue.clone());
        let observer_cache = cache_registry
            .slot::<TensorPreparationCache<T>>()
            .expect("tensor preparation cache slot initialized")
            .clone();
        command_buffer_pipeline::register_completion_observer(
            &command_queue,
            Arc::new(move |cmd_buffer: &CommandBuffer| {
                observer_cache.validate_states(cmd_buffer);
            }),
        );

        Ok(Context::<T> {
            device,
            command_queue,
            pool,
            kernel_manager: KernelManager::new(),
            backend_registry: KernelBackendRegistry::from_environment(),
            pooled_bytes_allocated: 0,
            pooled_allocations: 0,
            pool_resets: 0,
            rng_seed_counter: 0,
            cache_registry,
            active_cmd_buffer: None,
            command_buffer_pipeline: pipeline,
            active_resource_cache: None,
            sampler_buffers: SamplerBuffers::default(),
            forced_matmul_backend: forced_backend,
            mlx_kernel_cache: MlxKernelCache::default(),
            sdpa_workspaces: FxHashMap::default(),
            kv_repeat_workspaces: FxHashMap::default(),
            pending_gpu_scope: None,
            gpu_scope_stack: Vec::new(),
        })
    }

    #[inline]
    pub fn tensor_dtype(&self) -> Dtype {
        T::DTYPE
    }

    #[inline]
    pub fn backend_registry(&self) -> &KernelBackendRegistry {
        &self.backend_registry
    }

    #[inline]
    pub fn backend_registry_mut(&mut self) -> &mut KernelBackendRegistry {
        &mut self.backend_registry
    }

    #[inline]
    pub(crate) fn kv_cache_state(&self) -> &KvCacheState<T> {
        self.cache_registry
            .slot::<KvCacheState<T>>()
            .expect("KV cache state slot initialized")
    }

    #[inline]
    pub(crate) fn kv_cache_state_mut(&mut self) -> &mut KvCacheState<T> {
        self.cache_registry
            .slot_mut_existing::<KvCacheState<T>>()
            .expect("KV cache state slot initialized")
    }

    #[inline]
    pub(crate) fn tensor_preparation_cache(&self) -> &TensorPreparationCache<T> {
        self.cache_registry
            .slot::<TensorPreparationCache<T>>()
            .expect("tensor preparation cache slot initialized")
    }

    #[inline]
    pub fn kv_cache_pool(&self) -> &MemoryPool {
        self.kv_cache_state().pool()
    }

    #[inline]
    pub fn kv_cache_pool_mut(&mut self) -> &mut MemoryPool {
        self.kv_cache_state_mut().pool_mut()
    }

    #[inline]
    pub fn kv_caches(&self) -> &FxHashMap<usize, KvCacheEntry<T>> {
        self.kv_cache_state().caches()
    }

    #[inline]
    pub fn kv_caches_mut(&mut self) -> &mut FxHashMap<usize, KvCacheEntry<T>> {
        self.kv_cache_state_mut().caches_mut()
    }

    #[inline]
    pub fn apply_backend_overrides(&mut self, overrides: KernelBackendOverrides) {
        self.backend_registry.apply_overrides(overrides);
    }

    #[inline]
    pub fn set_global_backend_override(&mut self, override_policy: KernelBackendOverride) {
        self.backend_registry.set_global_override(override_policy);
    }

    pub fn set_sdpa_backend_override(&mut self, override_policy: KernelBackendOverride) {
        self.backend_registry.set_sdpa_override(override_policy);
    }

    #[inline]
    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.active_resource_cache.as_ref().map(|cache| cache.get_stats())
    }

    #[inline]
    pub fn clear_cache(&mut self) {
        if let Some(cache) = self.active_resource_cache.as_mut() {
            cache.clear();
        }
        self.sdpa_workspaces.clear();
        // Also clear tensor preparation cache when clearing other caches
        self.tensor_preparation_cache().clear();
    }

    #[inline]
    pub fn reset_pool(&mut self) {
        self.pool.reset();
    }

    /// Allocate a U32 tensor from the pool
    pub fn alloc_u32_tensor(&mut self, dims: Vec<usize>) -> Result<crate::tensor::Tensor<crate::tensor::U32>, MetalError> {
        let pooled_alloc = self.pool.alloc_tensor::<crate::tensor::U32>(dims)?;
        Ok(pooled_alloc.into_tensor())
    }

    #[inline]
    pub fn get_tensor_preparation_metrics(&self) -> crate::caching::TensorPreparationMetrics {
        self.tensor_preparation_cache().get_metrics()
    }

    #[inline]
    pub fn report_tensor_preparation_metrics(&self) {
        self.tensor_preparation_cache().report_metrics();
    }

    #[cfg(test)]
    pub(crate) fn force_enable_profiling_for_tests(&mut self) {
        crate::profiling_state::set_profiling_state(true);
    }

    pub(crate) fn ensure_active_cmd_buffer(&mut self) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer_internal(true)
    }

    #[inline]
    fn ensure_active_cmd_buffer_internal(&mut self, ensure_cache: bool) -> Result<(), MetalError> {
        if let Some(committed) = self.active_cmd_buffer.as_ref()
            && committed.is_committed()
        {
            let committed = self.active_cmd_buffer.take().expect("active command buffer must exist");
            let completions = self.wait_for_command_buffer(committed, None);
            self.process_pipeline_completions(completions);
        }

        if self.active_cmd_buffer.is_none() {
            let (mut cmd_buf, completed) = self.command_buffer_pipeline.acquire()?;
            if crate::profiling_state::get_profiling_state()
                && let Some(profiler) = metallic_instrumentation::gpu_profiler::GpuProfiler::attach(&cmd_buf, true)
            {
                cmd_buf.retain_profiler(profiler);
            }
            self.process_pipeline_completions(completed);
            self.active_cmd_buffer = Some(cmd_buf);
        }

        if ensure_cache && self.active_resource_cache.is_none() {
            self.active_resource_cache = Some(crate::caching::ResourceCache::with_device(self.device.clone()));
        }

        Ok(())
    }

    fn wait_for_command_buffer(
        &mut self,
        command_buffer: CommandBuffer,
        label: Option<super::utils::GpuProfilerLabel>,
    ) -> Vec<PipelineCompletion> {
        command_buffer_pipeline::wait_with_pipeline(&self.command_queue, &command_buffer, label)
    }

    pub(crate) fn active_command_buffer_mut(&mut self) -> Result<&mut crate::operation::CommandBuffer, MetalError> {
        self.ensure_active_cmd_buffer()?;
        Ok(self.active_cmd_buffer.as_mut().expect("active command buffer must exist"))
    }

    #[inline]
    pub(crate) fn active_command_buffer_mut_without_cache(&mut self) -> Result<&mut crate::operation::CommandBuffer, MetalError> {
        self.ensure_active_cmd_buffer_internal(false)?;
        Ok(self.active_cmd_buffer.as_mut().expect("active command buffer must exist"))
    }

    #[inline]
    pub(crate) fn mark_tensor_pending<U: TensorElement>(&self, tensor: &Tensor<U>) {
        tensor.mark_device_dirty();
        // Mark tensor as dirty in the preparation cache since it's now pending on GPU
        self.tensor_preparation_cache().mark_dirty(tensor);
        if let Some(active) = &self.active_cmd_buffer {
            tensor.defining_cmd_buffer.borrow_mut().replace(active.clone());
        }
    }

    pub(crate) fn call_with_cache<K: crate::kernels::DefaultKernelInvocable>(
        &mut self,
        args: K::Args<'_, T>,
        cache: &mut crate::caching::ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        // Get the current scope path to potentially nest with the kernel being called
        let current_scope_path = if let Some(current_label) = self.current_gpu_scope_label() {
            Some(current_label.op_name)
        } else {
            None
        };

        // Add operation name to current scope if we're already in a scope, creating nested path
        // Use the full type name in hot path - formatting to extract op name can be done in frontend
        let op_type_name = std::any::type_name::<K>();

        if let Some(mut current_path) = current_scope_path {
            // Append operation name to current scope path to create nested structure
            current_path.push('/');
            current_path.push_str(op_type_name);
            self.set_pending_gpu_scope(current_path);
        } else {
            // If not in any scope, just set the operation name as pending
            self.set_pending_gpu_scope(op_type_name);
        }

        self.ensure_active_cmd_buffer_internal(false)?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            Some(self.kernel_manager.get_pipeline(kernel_func, T::DTYPE, &self.device)?)
        } else {
            None
        };

        let (operation, output) = K::new(self, args, pipeline, Some(cache))?;

        let command_buffer = self.active_command_buffer_mut_without_cache()?;
        command_buffer.record(&*operation, cache)?;
        // Consume any pending GPU scope that was set for this call, since operations
        // may not always consume them (especially when nested scopes are created by call() itself)
        let _profiler_label = self.take_gpu_scope();

        debug_assert!(
            self.pending_gpu_scope.is_none(),
            "pending GPU scope should be consumed by kernel operation or by us"
        );
        self.mark_tensor_pending(&output);

        self.finalize_active_command_buffer_if_latency();

        Ok(output)
    }

    pub fn call_custom<K: CustomKernelInvocable>(
        &mut self,
        args: K::Args<'_, T>,
    ) -> Result<<K::OutputTuple<T> as MultiTensorOutput<T>>::Tensors, MetalError>
    where
        K::OutputTuple<T>: MultiTensorOutput<T>,
    {
        // Get the current scope path to potentially nest with the kernel being called
        let current_scope_path = if let Some(current_label) = self.current_gpu_scope_label() {
            Some(current_label.op_name)
        } else {
            None
        };

        // Add operation name to current scope if we're already in a scope, creating nested path
        // Use the full type name in hot path - formatting to extract op name can be done in frontend
        let op_type_name = std::any::type_name::<K>();

        if let Some(mut current_path) = current_scope_path {
            // Append operation name to current scope path to create nested structure
            current_path.push('/');
            current_path.push_str(op_type_name);
            self.set_pending_gpu_scope(current_path);
        } else {
            // If not in any scope, just set the operation name as pending
            self.set_pending_gpu_scope(op_type_name);
        }

        self.ensure_active_cmd_buffer()?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            // Use the input tensor dtype (T) for pipeline selection
            Some(self.kernel_manager.get_pipeline(kernel_func, T::DTYPE, &self.device)?)
        } else {
            None // For operations that don't need a pipeline
        };

        let mut cache = self
            .active_resource_cache
            .take()
            .expect("active resource cache must be initialized");

        let (operation, output) = K::new(self, args, pipeline, Some(&mut cache))?;

        if self.active_cmd_buffer.as_ref().map(|cb| cb.is_committed()).unwrap_or(false) {
            drop(cache);
            self.ensure_active_cmd_buffer()?;
            cache = self
                .active_resource_cache
                .take()
                .expect("active resource cache must be initialized after refresh");
        }

        if self.active_cmd_buffer.is_none() {
            // `K::new` may materialize resources that require a fresh command buffer,
            // so ensure one is available without reinitializing the resource cache we
            // already pulled above.
            self.ensure_active_cmd_buffer_internal(false)?;
        }

        let command_buffer = self.active_cmd_buffer.as_mut().expect("active command buffer must exist");

        command_buffer.record(&*operation, &mut cache)?;

        // Consume any pending GPU scope that was set for this call, since operations
        // may not always consume them (especially when nested scopes are created by call() itself)
        let _profiler_label = self.take_gpu_scope();

        debug_assert!(
            self.pending_gpu_scope.is_none(),
            "pending GPU scope should be consumed by kernel operation or by us"
        );

        self.active_resource_cache = Some(cache);

        // Mark all output tensors as pending
        <K::OutputTuple<T> as MultiTensorOutput<T>>::mark_pending(self, &output);

        self.finalize_active_command_buffer_if_latency();

        Ok(output)
    }

    pub fn call<K: DefaultKernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
        // Get the current scope path to potentially nest with the kernel being called
        let current_scope_path = if let Some(current_label) = self.current_gpu_scope_label() {
            Some(current_label.op_name)
        } else {
            None
        };

        // Add operation name to current scope if we're already in a scope, creating nested path
        // Use the full type name in hot path - formatting to extract op name can be done in frontend
        let op_type_name = std::any::type_name::<K>();

        if let Some(mut current_path) = current_scope_path {
            // Append operation name to current scope path to create nested structure
            current_path.push('/');
            current_path.push_str(op_type_name);
            self.set_pending_gpu_scope(current_path);
        } else {
            // If not in any scope, just set the operation name as pending
            self.set_pending_gpu_scope(op_type_name);
        }

        self.ensure_active_cmd_buffer()?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            Some(self.kernel_manager.get_pipeline(kernel_func, T::DTYPE, &self.device)?)
        } else {
            None // For MPS operations that don't need a pipeline
        };

        let mut cache = self
            .active_resource_cache
            .take()
            .expect("active resource cache must be initialized");

        let (operation, output) = K::new(self, args, pipeline, Some(&mut cache))?;

        if self.active_cmd_buffer.as_ref().map(|cb| cb.is_committed()).unwrap_or(false) {
            drop(cache);
            self.ensure_active_cmd_buffer()?;
            cache = self
                .active_resource_cache
                .take()
                .expect("active resource cache must be initialized after refresh");
        }

        if self.active_cmd_buffer.is_none() {
            // `K::new` may materialize resources that require a fresh command buffer,
            // so ensure one is available without reinitializing the resource cache we
            // already pulled above.
            self.ensure_active_cmd_buffer_internal(false)?;
        }

        let command_buffer = self.active_cmd_buffer.as_mut().expect("active command buffer must exist");

        command_buffer.record(&*operation, &mut cache)?;

        // Consume any pending GPU scope that was set for this call, since operations
        // may not always consume them (especially when nested scopes are created by call() itself)
        let _profiler_label = self.take_gpu_scope();

        debug_assert!(
            self.pending_gpu_scope.is_none(),
            "pending GPU scope should be consumed by kernel operation or by us"
        );

        self.active_resource_cache = Some(cache);

        self.mark_tensor_pending(&output);

        self.finalize_active_command_buffer_if_latency();

        Ok(output)
    }
}
