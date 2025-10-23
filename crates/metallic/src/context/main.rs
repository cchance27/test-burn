use kernels::{KernelBackendOverride, KernelBackendOverrides, KernelBackendRegistry, KernelManager, matmul_mlx::MlxKernelCache};
use metallic_instrumentation::config::AppConfig;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use rustc_hash::FxHashMap;

use crate::{
    Tensor, TensorElement, context::detect_forced_matmul_backend, error::MetalError, kernels, operation::CommandBuffer, pool::MemoryPool, profiling_state, resource_cache::{CacheStats, ResourceCache}, tensor::Dtype, tensor_preparation_cache::TensorPreparationCache
};

#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
}

pub struct KvWritePlan<T: TensorElement> {
    pub k_src: Tensor<T>,
    pub v_src: Tensor<T>,
    pub k_cache: Tensor<T>,
    pub v_cache: Tensor<T>,
    pub canonical_heads: usize,
    pub repeated_heads: usize,
    pub group_size: usize,
    pub group_size_u32: u32,
    pub seq_in_src: usize,
    pub head_dim: usize,
    pub capacity_seq_val: usize,
    pub element_size: usize,
    pub src_head_stride: u32,
    pub src_seq_stride: u32,
    pub dst_head_stride: u32,
    pub dst_seq_stride: u32,
    pub total_threads: u32,
    pub heads_u32: u32,
    pub head_dim_u32: u32,
    pub seq_len_u32: u32,
    pub step_u32: u32,
    pub step: usize,
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

#[derive(Clone)]
pub(crate) struct KvCacheEntry<T: TensorElement> {
    pub k: Tensor<T>,
    pub v: Tensor<T>,
    #[allow(dead_code)]
    pub dtype: Dtype,
    pub element_size: usize,
    pub zeroing_complete: bool,
    pub capacity: usize,
}

impl<T: TensorElement> KvCacheEntry<T> {
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.k.size_bytes() + self.v.size_bytes()
    }
}

const KV_CACHE_POOL_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB

pub struct Context<T: TensorElement> {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
    pub kv_cache_pool: MemoryPool,
    pub kernel_manager: KernelManager,
    backend_registry: KernelBackendRegistry,
    // Metrics counters
    pub pooled_bytes_allocated: usize,
    pub pooled_allocations: usize,
    pub pool_resets: usize,
    // RNG seed counter for deterministic random generation
    pub rng_seed_counter: u64,

    // Per-layer on-device KV caches stored centrally for developer DX.
    pub(crate) kv_caches: FxHashMap<usize, KvCacheEntry<T>>,
    pub(crate) kv_cache_total_bytes: usize,

    /// Lazily created command buffer used to batch kernel dispatches until synchronization.
    pub(crate) active_cmd_buffer: Option<CommandBuffer>,
    /// Resource cache associated with the active command buffer.
    pub(crate) active_resource_cache: Option<ResourceCache>,
    /// Workspace reused across sampling invocations to avoid per-token allocations.
    pub sampler_buffers: SamplerBuffers,
    /// Optional override for the matmul backend chosen by this context.
    pub(crate) forced_matmul_backend: super::utils::MatMulBackendOverride,
    /// Cache of MLX compute pipelines keyed by kernel configuration.
    pub(crate) mlx_kernel_cache: MlxKernelCache,
    pub(crate) sdpa_workspaces: FxHashMap<super::sdpa_workspace::SdpaWorkspaceKey, super::sdpa_workspace::SdpaWorkspaceState>,
    pub(crate) kv_repeat_workspaces: FxHashMap<RepeatKvWorkspaceKey, RepeatKvWorkspaceEntry<T>>,
    pub(crate) tensor_preparation_cache: TensorPreparationCache<T>,
    pub(crate) pending_gpu_scope: Option<super::utils::GpuProfilerLabel>,
    pub(crate) gpu_scope_stack: Vec<super::utils::GpuProfilerLabel>,
}

impl<T: TensorElement> Context<T> {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device, &command_queue)?;
        let kv_cache_pool = MemoryPool::with_limit(&device, &command_queue, KV_CACHE_POOL_MAX_BYTES)?;
        let forced_backend = detect_forced_matmul_backend();
        // Initialize the global profiling state based on environment configuration
        profiling_state::initialize_profiling_state_from_env();

        if AppConfig::profiling_forced() {
            profiling_state::set_profiling_state(true);
        }

        Ok(Context::<T> {
            device,
            command_queue,
            pool,
            kv_cache_pool,
            kernel_manager: KernelManager::new(),
            backend_registry: KernelBackendRegistry::from_environment(),
            pooled_bytes_allocated: 0,
            pooled_allocations: 0,
            pool_resets: 0,
            rng_seed_counter: 0,
            kv_caches: FxHashMap::default(),
            kv_cache_total_bytes: 0,
            active_cmd_buffer: None,
            active_resource_cache: None,
            sampler_buffers: SamplerBuffers::default(),
            forced_matmul_backend: forced_backend,
            mlx_kernel_cache: MlxKernelCache::default(),
            sdpa_workspaces: FxHashMap::default(),
            kv_repeat_workspaces: FxHashMap::default(),
            tensor_preparation_cache: TensorPreparationCache::new(),
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
    pub fn apply_backend_overrides(&mut self, overrides: KernelBackendOverrides) {
        self.backend_registry.apply_overrides(overrides);
    }

    #[inline]
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
        self.tensor_preparation_cache.clear();
    }

    #[inline]
    pub fn reset_pool(&mut self) {
        self.pool.reset();
    }

    #[inline]
    pub fn get_tensor_preparation_metrics(&self) -> crate::tensor_preparation_cache::TensorPreparationMetrics {
        self.tensor_preparation_cache.get_metrics()
    }

    #[inline]
    pub fn report_tensor_preparation_metrics(&self) {
        self.tensor_preparation_cache.report_metrics();
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
        // Check and refresh committed command buffer if needed
        if let Some(active) = self.active_cmd_buffer.as_ref()
            && active.is_committed()
        {
            if !active.is_completed() {
                active.wait();
            }
            // Buffer is committed, clear it to create a fresh one
            self.active_cmd_buffer = None;
        }

        // Check if we need to create a new buffer
        let new_buffer_created = if self.active_cmd_buffer.is_none() {
            let cmd_buf = crate::operation::CommandBuffer::new(&self.command_queue)?;
            if crate::profiling_state::get_profiling_state()
                && let Some(profiler) = metallic_instrumentation::gpu_profiler::GpuProfiler::attach(&cmd_buf, true)
            {
                cmd_buf.retain_profiler(profiler);
            }
            self.active_cmd_buffer = Some(cmd_buf);
            true
        } else {
            false
        };

        if ensure_cache && self.active_resource_cache.is_none() {
            self.active_resource_cache = Some(crate::resource_cache::ResourceCache::with_device(self.device.clone()));
        }

        // If a new command buffer was created, we need to clear preparation states
        // for tensors prepared for the previous command buffer
        if new_buffer_created {
            // No need to explicitly clear preparation states here since the preparation
            // cache will detect command buffer changes and handle invalidation appropriately
        }

        Ok(())
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
    pub(crate) fn mark_tensor_pending(&self, tensor: &Tensor<T>) {
        tensor.mark_device_dirty();
        // Mark tensor as dirty in the preparation cache since it's now pending on GPU
        self.tensor_preparation_cache.mark_dirty(tensor);
        if let Some(active) = &self.active_cmd_buffer {
            tensor.defining_cmd_buffer.borrow_mut().replace(active.clone());
        }
    }

    pub(crate) fn call_with_cache<K: crate::kernels::KernelInvocable>(
        &mut self,
        args: K::Args<'_, T>,
        cache: &mut crate::resource_cache::ResourceCache,
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

    pub fn call<K: crate::kernels::KernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
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
