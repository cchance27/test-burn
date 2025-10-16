use super::error::MetalError;
use super::operation::CommandBuffer;
use super::pool::MemoryPool;
use super::profiling_state;
use super::resource_cache::{CacheStats, ResourceCache};
use crate::kernels::elemwise_add::BroadcastElemwiseAddInplaceOp;
use crate::tensor::Dtype;
use crate::{Tensor, TensorElement, cache_keys::SdpaKey, kernels};
use kernels::kv_cache_write::{KvCacheWriteConfig, KvCacheWriteOp};
use kernels::matmul_gemv::MatmulGemvOp;
use kernels::matmul_mlx::{MatMulMlxOp, MlxKernelCache};
use kernels::matmul_mps::{MatMulBackend, MatMulMpsAlphaBetaOp, MatMulMpsOp};
use kernels::scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp;
use kernels::{KernelInvocable, KernelManager};
use metallic_env::FORCE_MATMUL_BACKEND_VAR;
use metallic_instrumentation::record_metric_async;
use metallic_instrumentation::{MetricEvent, config::AppConfig, gpu_profiler::GpuProfiler};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryUsage {
    pub pool_used: usize,
    pub pool_capacity: usize,
    pub kv_used: usize,
    pub kv_capacity: usize,
    pub kv_cache_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MatmulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct GpuProfilerLabel {
    pub op_name: String,
    pub backend: String,
}

impl GpuProfilerLabel {
    pub fn new(op_name: String, backend: String) -> Self {
        Self { op_name, backend }
    }

    pub fn fallback(op_name: &str) -> Self {
        Self {
            op_name: op_name.to_string(),
            backend: GPU_PROFILER_BACKEND.to_string(),
        }
    }
}

pub(crate) const GPU_PROFILER_BACKEND: &str = "Metal";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMulBackendOverride {
    Default,
    Force(MatMulBackend),
    Auto,
}

fn detect_forced_matmul_backend() -> MatMulBackendOverride {
    match FORCE_MATMUL_BACKEND_VAR.get() {
        Ok(Some(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                MatMulBackendOverride::Default
            } else {
                match trimmed.to_ascii_lowercase().as_str() {
                    "mlx" => MatMulBackendOverride::Force(MatMulBackend::Mlx),
                    "mps" => MatMulBackendOverride::Force(MatMulBackend::Mps),
                    "gemv" => MatMulBackendOverride::Force(MatMulBackend::Gemv),
                    "auto" => MatMulBackendOverride::Auto,
                    _ => MatMulBackendOverride::Default,
                }
            }
        }
        _ => MatMulBackendOverride::Default,
    }
}

#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
}

struct KvWritePlan<T: TensorElement> {
    k_src: Tensor<T>,
    v_src: Tensor<T>,
    k_cache: Tensor<T>,
    v_cache: Tensor<T>,
    canonical_heads: usize,
    repeated_heads: usize,
    group_size: usize,
    group_size_u32: u32,
    seq_in_src: usize,
    head_dim: usize,
    capacity_seq_val: usize,
    element_size: usize,
    src_head_stride: u32,
    src_seq_stride: u32,
    dst_head_stride: u32,
    dst_seq_stride: u32,
    total_threads: u32,
    heads_u32: u32,
    head_dim_u32: u32,
    seq_len_u32: u32,
    step_u32: u32,
    step: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SdpaWorkspaceKey {
    buffer: usize,
    offset: usize,
}

impl SdpaWorkspaceKey {
    fn from_tensor<T: TensorElement>(tensor: &Tensor<T>) -> Self {
        let buffer = Retained::as_ptr(&tensor.buf) as *const _ as usize;
        Self {
            buffer,
            offset: tensor.offset,
        }
    }
}

#[derive(Clone, Debug)]
struct SdpaWorkspaceState {
    descriptor: SdpaKey,
    last_seq_q: usize,
    last_seq_k: usize,
}

impl SdpaWorkspaceState {
    fn new(descriptor: SdpaKey) -> Self {
        Self {
            descriptor,
            last_seq_q: 0,
            last_seq_k: 0,
        }
    }

    fn reset(&mut self, descriptor: SdpaKey) {
        self.descriptor = descriptor;
        self.last_seq_q = 0;
        self.last_seq_k = 0;
    }
}

/// The main context for Metal operations.
pub struct Context<T: TensorElement> {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
    pub kv_cache_pool: MemoryPool,
    pub kernel_manager: KernelManager,
    // Metrics counters
    pub pooled_bytes_allocated: usize,
    pub pooled_allocations: usize,
    pub pool_resets: usize,
    // RNG seed counter for deterministic random generation
    pub rng_seed_counter: u64,

    // Per-layer on-device KV caches stored centrally for developer DX.
    pub(crate) kv_caches: FxHashMap<usize, KvCacheEntry<T>>,
    kv_cache_total_bytes: usize,

    /// Lazily created command buffer used to batch kernel dispatches until synchronization.
    active_cmd_buffer: Option<CommandBuffer>,
    /// Resource cache associated with the active command buffer.
    active_resource_cache: Option<ResourceCache>,
    /// Workspace reused across sampling invocations to avoid per-token allocations.
    pub sampler_buffers: SamplerBuffers,
    /// Optional override for the matmul backend chosen by this context.
    forced_matmul_backend: MatMulBackendOverride,
    /// Cache of MLX compute pipelines keyed by kernel configuration.
    pub(crate) mlx_kernel_cache: MlxKernelCache,
    sdpa_workspaces: FxHashMap<SdpaWorkspaceKey, SdpaWorkspaceState>,
    pending_gpu_scope: Option<GpuProfilerLabel>,
    gpu_scope_stack: Vec<GpuProfilerLabel>,
}

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
    fn total_bytes(&self) -> usize {
        self.k.size_bytes() + self.v.size_bytes()
    }
}

const KV_CACHE_POOL_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB

impl<T: TensorElement> Context<T> {
    /// Synchronize pending GPU work, committing and waiting on the active command buffer.
    /// Falls back to the legacy submit/wait path if no active buffer exists.
    pub fn synchronize(&mut self) {
        if let Some(cmd_buf) = self.active_cmd_buffer.take() {
            let wait_start = std::time::Instant::now();
            cmd_buf.commit();
            cmd_buf.wait();
            let waited = wait_start.elapsed();
            if !waited.is_zero() {
                if let Some(label) = self.current_gpu_scope_label() {
                    let path = label.op_name;
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/cb_wait", path),
                        backend: label.backend,
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                } else {
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: "Generation Loop/cb_wait".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                }
            }
            return;
        }

        if let Some(cb) = self.command_queue.commandBuffer() {
            // No active CB; attribute to the outermost scope if present
            let wait_start = std::time::Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            let waited = wait_start.elapsed();
            if !waited.is_zero() {
                if let Some(label) = self.current_gpu_scope_label() {
                    let path = label.op_name;
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/cb_wait", path),
                        backend: label.backend,
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                } else {
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: "Generation Loop/cb_wait".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                }
            }
        }
    }

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
            pending_gpu_scope: None,
            gpu_scope_stack: Vec::new(),
        })
    }

    #[inline]
    pub fn tensor_dtype(&self) -> Dtype {
        T::DTYPE
    }

    pub fn call<K: KernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
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

    pub(crate) fn finalize_active_command_buffer_if_latency(&mut self) {
        if crate::profiling_state::get_profiling_state()
            && let Some(cmd_buf) = self.active_cmd_buffer.take()
        {
            // Attribute CB finalize commit/wait to the current scope to avoid 'Other'
            let wait_start = std::time::Instant::now();
            cmd_buf.commit();
            cmd_buf.wait();
            let waited = wait_start.elapsed();
            if !waited.is_zero() {
                if let Some(label) = self.current_gpu_scope_label() {
                    record_metric_async!(MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/cb_wait", label.op_name),
                        backend: label.backend,
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                } else {
                    record_metric_async!(MetricEvent::GpuOpCompleted {
                        op_name: "Generation Loop/cb_wait".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                }
            }
        }
    }

    #[inline]
    pub fn set_pending_gpu_scope<S: Into<String>>(&mut self, op_name: S) {
        self.pending_gpu_scope = Some(GpuProfilerLabel::new(op_name.into(), GPU_PROFILER_BACKEND.to_string()));
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

    // Device capability hints used by dispatcher heuristics. These are conservative defaults
    // and should be updated to query real device/feature sets.
    #[inline]
    pub fn device_has_simdgroup_mm(&self) -> bool {
        // TODO(DEBT): Detect simdgroup matrix multiply support via Metal feature sets / GPU family.
        // For now, return false; dispatcher thresholds can be tuned via env.
        false
    }

    #[inline]
    pub fn max_threads_per_threadgroup(&self) -> usize {
        // TODO(DEBT): Provide a pipeline-aware value; fall back to a conservative upper bound.
        1024
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

    // Compute matmul dims from tensor views
    fn compute_matmul_dims(&self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<MatmulDims, MetalError> {
        let a_view = a.as_mps_matrix_batch_view()?;
        let b_view = b.as_mps_matrix_batch_view()?;
        let (a_rows, a_cols) = if transpose_a {
            (a_view.columns, a_view.rows)
        } else {
            (a_view.rows, a_view.columns)
        };
        let (b_rows, b_cols) = if transpose_b {
            (b_view.columns, b_view.rows)
        } else {
            (b_view.rows, b_view.columns)
        };
        if a_cols != b_rows {
            return Err(MetalError::InvalidOperation("matmul dims mismatch".into()));
        }
        Ok(MatmulDims {
            batch: a_view.batch.max(b_view.batch),
            m: a_rows,
            n: b_cols,
            k: a_cols,
        })
    }

    fn should_use_mlx_bias(&self, dims: &MatmulDims) -> bool {
        if dims.n <= 32 {
            return false;
        }

        if dims.batch == 1 && dims.m <= 4 {
            // For very skinny decode projections benchmark data shows MLX holds an advantage
            // unless both the output width is extremely small and the reduction dim dwarfs it.
            if dims.n <= 1024 && dims.k >= dims.n {
                return false;
            }
        }

        if dims.m >= 1024 || dims.n >= 1024 {
            return true;
        }

        if dims.batch > 1 && dims.m >= 256 && dims.n >= 256 {
            return true;
        }

        false
    }

    #[inline]
    fn has_strided_mps_batch(&self, tensors: &[&Tensor<T>]) -> bool {
        tensors.iter().any(|tensor| {
            tensor
                .as_mps_matrix_batch_view()
                .map(|view| view.batch > 1 && view.matrix_bytes != view.rows * view.row_bytes)
                .unwrap_or(false)
        })
    }

    fn should_use_mlx_dense(&self, dims: &MatmulDims, has_strided_batch: bool) -> bool {
        if dims.n <= 32 && !has_strided_batch {
            return false;
        }

        if dims.batch == 1 && dims.m <= 4 {
            if !has_strided_batch && dims.n <= 128 && dims.k >= dims.n * 2 {
                return false;
            }
            if !has_strided_batch {
                let four_k = dims.k.saturating_mul(4);
                let four_n = dims.n.saturating_mul(4);
                if dims.n >= four_k || dims.k >= four_n {
                    return false;
                }
            }
        }

        if dims.m >= 1024 || dims.n >= 1024 {
            return true;
        }

        true
    }

    #[inline]
    fn can_use_gemv(&self, dims: &MatmulDims, transpose_a: bool, transpose_b: bool) -> bool {
        if transpose_a || transpose_b {
            return false;
        }

        if dims.batch != 1 {
            return false;
        }

        dims.m == 1
    }

    #[inline]
    pub fn matmul(&mut self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b)),
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.call::<MatmulGemvOp>((a, b))
                            } else {
                                self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            }
                        }
                        Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                    }
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.call::<MatmulGemvOp>((a, b));
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))
                        }
                    }
                    Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                }
            }
        }
    }

    #[inline]
    pub(crate) fn matmul_with_cache(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => self.call_with_cache::<MatMulMpsOp>((a, b, transpose_a, transpose_b), cache),
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.call_with_cache::<MatmulGemvOp>((a, b), cache)
                            } else {
                                self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            }
                        }
                        Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                    }
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.call_with_cache::<MatmulGemvOp>((a, b), cache);
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            self.call_with_cache::<MatMulMpsOp>((a, b, transpose_a, transpose_b), cache)
                        }
                    }
                    Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn fused_qkv_projection(
        &mut self,
        x_flat: &Tensor<T>,
        fused_weight: &Tensor<T>,
        fused_bias: &Tensor<T>,
        d_model: usize,
        kv_dim: usize,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        let x_dims = x_flat.dims();
        if x_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "fused_qkv_projection expects a 2D input [m, d_model], got {:?}",
                x_dims
            )));
        }

        let _m = x_dims[0];
        let in_features = x_dims[1];
        if in_features != d_model {
            return Err(MetalError::InvalidShape(format!(
                "Input feature size {} does not match d_model {}",
                in_features, d_model
            )));
        }

        let weight_dims = fused_weight.dims();
        if weight_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Fused weight must be 2D [d_model, qkv], got {:?}",
                weight_dims
            )));
        }

        let expected_total = d_model + 2 * kv_dim;
        if weight_dims[0] != d_model || weight_dims[1] != expected_total {
            return Err(MetalError::InvalidShape(format!(
                "Fused weight dims {:?} incompatible with d_model {} and kv_dim {}",
                weight_dims, d_model, kv_dim
            )));
        }

        if fused_bias.dims() != [expected_total] {
            return Err(MetalError::InvalidShape(format!(
                "Fused bias dims {:?} incompatible with expected total {}",
                fused_bias.dims(),
                expected_total
            )));
        }

        let linear = self.matmul_bias_add(x_flat, fused_weight, fused_bias, false, false)?;

        let q_range_end = d_model;
        let k_range_end = d_model + kv_dim;
        let v_range_end = expected_total;

        let q_out = linear.slice_last_dim(0..q_range_end)?;
        let k_out = linear.slice_last_dim(d_model..k_range_end)?;
        let v_out = linear.slice_last_dim(k_range_end..v_range_end)?;

        Ok((q_out, k_out, v_out))
    }

    #[inline]
    pub fn matmul_bias_add(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        bias: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, Some(bias), None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => {
                    let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                    linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                    Ok(linear)
                }
                MatMulBackend::Gemv => {
                    let mut linear = self.call::<MatmulGemvOp>((a, b))?;
                    linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                    Ok(linear)
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        let use_mlx = self.should_use_mlx_bias(&dimensions);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, Some(bias), None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                            linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                            Ok(linear)
                        }
                    }
                    Err(_) => {
                        let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                        linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                        Ok(linear)
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn matmul_alpha_beta(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        result: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),

                MatMulBackend::Mps => self.call::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta)),

                MatMulBackend::Gemv => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (_, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                } else {
                    self.call::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta))
                }
            }
        }
    }

    pub(crate) fn call_with_cache<K: KernelInvocable>(
        &mut self,
        args: K::Args<'_, T>,
        cache: &mut ResourceCache,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_alpha_beta_with_cache(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        result: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
                MatMulBackend::Mps => {
                    self.call_with_cache::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta), cache)
                }
                MatMulBackend::Gemv => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (_, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                } else {
                    self.call_with_cache::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta), cache)
                }
            }
        }
    }

    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.active_resource_cache.as_ref().map(|cache| cache.get_stats())
    }

    pub fn clear_cache(&mut self) {
        if let Some(cache) = self.active_resource_cache.as_mut() {
            cache.clear();
        }
        self.sdpa_workspaces.clear();
    }

    pub fn reset_pool(&mut self) {
        self.pool.reset();
    }

    pub(crate) fn clear_kv_caches(&mut self) {
        self.kv_caches.clear();
        self.kv_cache_total_bytes = 0;
        self.sdpa_workspaces.clear();
    }

    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [batch * n_heads, seq_len, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_kv_cache(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        repeated_batch_heads: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        let repeated_dims = vec![repeated_batch_heads, seq_len, head_dim];

        // Allocate K and V tensors directly from the dedicated KV cache pool
        let k_allocation = self.kv_cache_pool.alloc_tensor::<T>(repeated_dims.clone())?;
        let v_allocation = self.kv_cache_pool.alloc_tensor::<T>(repeated_dims)?;

        let dtype = k_allocation.dtype();
        let element_size = k_allocation.element_size();
        debug_assert_eq!(dtype, v_allocation.dtype());
        let k = k_allocation.into_tensor();
        let v = v_allocation.into_tensor();

        // Manually zero the tensors using a blit command
        let k_size = k.size_bytes();
        let v_size = v.size_bytes();

        self.ensure_active_cmd_buffer()?;
        let cmd_buf = self.active_command_buffer_mut()?;
        {
            let encoder = cmd_buf.get_blit_encoder()?;
            encoder.fillBuffer_range_value(&k.buf, (k.offset..k.offset + k_size).into(), 0);
            encoder.fillBuffer_range_value(&v.buf, (v.offset..v.offset + v_size).into(), 0);
        }

        self.mark_tensor_pending(&k);
        self.mark_tensor_pending(&v);
        self.finalize_active_command_buffer_if_latency();

        let entry = KvCacheEntry {
            k,
            v,
            dtype,
            element_size,
            zeroing_complete: true,
            capacity: seq_len,
        };
        let entry_bytes = entry.total_bytes();
        if let Some(prev) = self.kv_caches.insert(layer_idx, entry) {
            self.kv_cache_total_bytes = self
                .kv_cache_total_bytes
                .saturating_sub(prev.total_bytes())
                .saturating_add(entry_bytes);
        } else {
            self.kv_cache_total_bytes = self.kv_cache_total_bytes.saturating_add(entry_bytes);
        }
        Ok(())
    }

    /// Write a single timestep of K and V (per-head flattened) into the per-layer cache at index `step`.
    /// - `k_step` and `v_step` must be contiguous tensors with shape [batch_heads, head_dim] or [batch_heads, 1, head_dim].
    ///   This performs a device blit copy from the source buffer into the cache at the correct offset.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        group_size: usize,
        k_step: &Tensor<T>,
        v_step: &Tensor<T>,
    ) -> Result<(), MetalError> {
        let plan = self.build_kv_write_plan(layer_idx, step, group_size, k_step, v_step)?;
        self.dispatch_kv_write(layer_idx, plan)
    }

    fn build_kv_write_plan(
        &self,
        layer_idx: usize,
        step: usize,
        group_size: usize,
        k_step: &Tensor<T>,
        v_step: &Tensor<T>,
    ) -> Result<KvWritePlan<T>, MetalError> {
        let k_src = k_step.clone();
        let v_src = v_step.clone();

        let dims = k_src.dims().to_vec();
        let (bh, seq_in_src, hd) = match dims.len() {
            2 => (dims[0], 1, dims[1]),
            3 => (dims[0], dims[1], dims[2]),
            _ => {
                return Err(MetalError::InvalidShape("write_kv_step expects source tensor rank 2 or 3".into()));
            }
        };

        let v_dims = v_src.dims().to_vec();
        let (v_bh, v_seq_in_src, v_hd) = match v_dims.len() {
            2 => (v_dims[0], 1, v_dims[1]),
            3 => (v_dims[0], v_dims[1], v_dims[2]),
            _ => {
                return Err(MetalError::InvalidShape("write_kv_step expects V tensor rank 2 or 3".into()));
            }
        };

        if seq_in_src != 1 {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step currently expects a single timestep in the source tensor".into(),
            ));
        }

        if v_seq_in_src != seq_in_src {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step expects matching sequence dims for K and V".into(),
            ));
        }

        let entry = self
            .kv_caches
            .get(&layer_idx)
            .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not allocated", layer_idx)))?;
        let k_cache = entry.k.clone();
        let v_cache = entry.v.clone();
        let capacity_seq_val = entry.capacity;
        let element_size = entry.element_size;

        if group_size == 0 {
            return Err(MetalError::InvalidOperation("write_kv_step requires a non-zero group size".into()));
        }

        if step >= capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Step {} exceeds KV cache capacity {} for layer {}",
                step, capacity_seq_val, layer_idx
            )));
        }

        let cache_dims = k_cache.dims();
        if cache_dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "KV cache tensor must have shape [batch_heads, seq_len, head_dim]".into(),
            ));
        }

        let expected_repeated_heads = cache_dims[0];
        let expected_hd = cache_dims[2];

        if hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_hd,
                actual: hd,
            });
        }

        if v_hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_hd,
                actual: v_hd,
            });
        }

        if bh != v_bh {
            return Err(MetalError::DimensionMismatch {
                expected: bh,
                actual: v_bh,
            });
        }

        let canonical_heads = bh;
        let repeated_heads_expected = canonical_heads
            .checked_mul(group_size)
            .ok_or_else(|| MetalError::InvalidOperation("group_size overflow while expanding KV heads".into()))?;

        if expected_repeated_heads != repeated_heads_expected {
            return Err(MetalError::DimensionMismatch {
                expected: repeated_heads_expected,
                actual: expected_repeated_heads,
            });
        }

        if step + seq_in_src > capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Writing KV step {} ({} timesteps) exceeds cache capacity {} for layer {}",
                step, seq_in_src, capacity_seq_val, layer_idx
            )));
        }

        if k_src.strides.len() != v_src.strides.len()
            || k_src.strides.first() != v_src.strides.first()
            || (k_src.strides.len() > 1 && k_src.strides[1] != v_src.strides[1])
        {
            return Err(MetalError::InvalidShape(
                "write_kv_step requires K and V to share the same layout".into(),
            ));
        }

        let head_dim_u32 = u32::try_from(expected_hd).map_err(|_| MetalError::InvalidShape("head dimension exceeds u32::MAX".into()))?;
        let heads_u32 = u32::try_from(canonical_heads).map_err(|_| MetalError::InvalidShape("batch_heads exceeds u32::MAX".into()))?;
        let seq_len_u32 = u32::try_from(seq_in_src).map_err(|_| MetalError::InvalidShape("sequence length exceeds u32::MAX".into()))?;
        let step_u32 = u32::try_from(step).map_err(|_| MetalError::InvalidShape("step index exceeds u32::MAX".into()))?;
        let src_head_stride =
            u32::try_from(k_src.strides[0]).map_err(|_| MetalError::InvalidShape("source head stride exceeds u32::MAX".into()))?;
        let src_seq_stride = if dims.len() == 3 {
            u32::try_from(k_src.strides[1]).map_err(|_| MetalError::InvalidShape("source sequence stride exceeds u32::MAX".into()))?
        } else {
            0
        };
        let dst_head_stride =
            u32::try_from(k_cache.strides[0]).map_err(|_| MetalError::InvalidShape("cache head stride exceeds u32::MAX".into()))?;
        let dst_seq_stride =
            u32::try_from(k_cache.strides[1]).map_err(|_| MetalError::InvalidShape("cache sequence stride exceeds u32::MAX".into()))?;

        let repeated_heads = expected_repeated_heads;
        let group_size_u32 = u32::try_from(group_size).map_err(|_| MetalError::InvalidShape("group size exceeds u32::MAX".into()))?;

        if !k_src.offset.is_multiple_of(element_size)
            || !v_src.offset.is_multiple_of(element_size)
            || k_cache.offset % element_size != 0
            || v_cache.offset % element_size != 0
        {
            return Err(MetalError::InvalidOperation("KV tensors must be element-aligned".into()));
        }

        let total_threads = heads_u32
            .checked_mul(head_dim_u32)
            .ok_or_else(|| MetalError::InvalidShape("thread count exceeds u32::MAX".into()))?;

        Ok(KvWritePlan {
            k_src,
            v_src,
            k_cache,
            v_cache,
            canonical_heads,
            repeated_heads,
            group_size,
            group_size_u32,
            seq_in_src,
            head_dim: expected_hd,
            capacity_seq_val,
            element_size,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            heads_u32,
            head_dim_u32,
            seq_len_u32,
            step_u32,
            step,
        })
    }

    fn dispatch_kv_write(&mut self, layer_idx: usize, plan: KvWritePlan<T>) -> Result<(), MetalError> {
        let KvWritePlan {
            k_src,
            v_src,
            k_cache,
            v_cache,
            canonical_heads,
            repeated_heads,
            group_size,
            group_size_u32,
            seq_in_src,
            head_dim,
            capacity_seq_val,
            element_size,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            heads_u32,
            head_dim_u32,
            seq_len_u32,
            step_u32,
            step,
        } = plan;

        let tensors: Vec<&Tensor<T>> = vec![&k_cache, &v_cache, &k_src, &v_src];
        self.prepare_tensors_for_active_cmd(&tensors)?;

        let config = KvCacheWriteConfig {
            canonical_heads: heads_u32,
            head_dim: head_dim_u32,
            seq_len: seq_len_u32,
            step: step_u32,
            group_size: group_size_u32,
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
            repeated_heads: u32::try_from(repeated_heads)
                .map_err(|_| MetalError::InvalidShape("repeated head count exceeds u32::MAX".into()))?,
        };

        match self.call::<KvCacheWriteOp>((k_src.clone(), v_src.clone(), k_cache.clone(), v_cache.clone(), config.clone())) {
            Ok(_) => {
                // Record metric for successful KV cache kernel dispatch using new instrumentation
                record_metric_async!(MetricEvent::GpuKernelDispatched {
                    kernel_name: "kv_cache_write".to_string(),
                    op_name: format!("kv_cache_write_step_{}_layer_{}", step, layer_idx),
                    thread_groups: (config.total_threads, 1, 1),
                });
            }
            Err(err) if Self::kv_cache_kernel_unavailable(&err) => {
                // Record metric for fallback blit operation using new instrumentation
                record_metric_async!(MetricEvent::GpuKernelDispatched {
                    kernel_name: "kv_cache_fallback_blit".to_string(),
                    op_name: format!("kv_cache_blit_step_{}_layer_{}", step, layer_idx),
                    thread_groups: (1, 1, 1), // Placeholder - actual thread groups would be computed differently for blit
                });
                return self.blit_write_kv_step(
                    layer_idx,
                    step,
                    &k_src,
                    &v_src,
                    &k_cache,
                    &v_cache,
                    seq_in_src,
                    canonical_heads,
                    head_dim,
                    capacity_seq_val,
                    element_size,
                    group_size,
                    repeated_heads,
                );
            }
            Err(err) => return Err(err),
        }

        if let Some(entry) = self.kv_caches.get_mut(&layer_idx) {
            entry.zeroing_complete = false;
        }

        self.mark_tensor_pending(&k_cache);
        self.mark_tensor_pending(&v_cache);
        self.finalize_active_command_buffer_if_latency();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn blit_write_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        k_src: &Tensor<T>,
        v_src: &Tensor<T>,
        k_cache: &Tensor<T>,
        v_cache: &Tensor<T>,
        seq_in_src: usize,
        canonical_heads: usize,
        head_dim: usize,
        capacity: usize,
        element_size: usize,
        group_size: usize,
        repeated_heads: usize,
    ) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer()?;
        let profiler_label = self
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("kv_cache_blit_op"));
        let cmd_buf = self.active_command_buffer_mut()?;
        let raw_cmd = cmd_buf.raw();
        let encoder = cmd_buf.get_blit_encoder()?;
        let _scope = GpuProfiler::profile_blit(raw_cmd, &encoder, profiler_label.op_name, profiler_label.backend);

        let cache_stride_elems = capacity * head_dim;
        let copy_bytes = head_dim * element_size;
        let k_dims_len = k_src.dims().len();
        let v_dims_len = v_src.dims().len();

        if repeated_heads != canonical_heads * group_size {
            return Err(MetalError::DimensionMismatch {
                expected: canonical_heads * group_size,
                actual: repeated_heads,
            });
        }

        unsafe {
            for head_idx in 0..canonical_heads {
                let src_elem_index = if k_dims_len == 2 {
                    head_idx * head_dim
                } else {
                    (head_idx * seq_in_src) * head_dim
                };
                let v_src_elem_index = if v_dims_len == 2 {
                    head_idx * head_dim
                } else {
                    (head_idx * seq_in_src) * head_dim
                };
                let src_offset_k = k_src.offset + src_elem_index * element_size;
                let src_offset_v = v_src.offset + v_src_elem_index * element_size;

                for group in 0..group_size {
                    let repeated_head = head_idx * group_size + group;
                    let dst_elem_index = repeated_head * cache_stride_elems + step * head_dim;

                    let dst_offset_k = k_cache.offset + dst_elem_index * element_size;
                    let dst_offset_v = v_cache.offset + dst_elem_index * element_size;

                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &k_src.buf,
                        src_offset_k,
                        &k_cache.buf,
                        dst_offset_k,
                        copy_bytes,
                    );
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &v_src.buf,
                        src_offset_v,
                        &v_cache.buf,
                        dst_offset_v,
                        copy_bytes,
                    );
                }
            }
        }

        self.mark_tensor_pending(k_cache);
        self.mark_tensor_pending(v_cache);
        self.finalize_active_command_buffer_if_latency();

        if let Some(entry) = self.kv_caches.get_mut(&layer_idx) {
            entry.zeroing_complete = false;
        }

        Ok(())
    }

    fn kv_cache_kernel_unavailable(err: &MetalError) -> bool {
        matches!(
            err,
            MetalError::LibraryCompilationFailed(_)
                | MetalError::FunctionCreationFailed(_)
                | MetalError::PipelineCreationFailed
                | MetalError::ComputeEncoderCreationFailed
                | MetalError::UnsupportedDtype { .. }
        )
    }

    /// Create a strided view of the KV cache exposing the first `active_steps` positions in
    /// [batch_heads, steps, head_dim] order while preserving the underlying cache stride.
    pub fn kv_cache_history_view(&mut self, cache: &Tensor<T>, active_steps: usize) -> Result<(Tensor<T>, usize), MetalError> {
        let dims = cache.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "KV cache tensor must have shape [batch_heads, seq_len, head_dim]".to_string(),
            ));
        }

        if active_steps == 0 || active_steps > dims[1] {
            return Err(MetalError::InvalidShape(format!(
                "Requested {} KV steps exceeds cache capacity {}",
                active_steps, dims[1]
            )));
        }

        let mut view = cache.clone();
        view.dims = vec![dims[0], active_steps, dims[2]];
        view.strides = vec![cache.strides[0], cache.strides[1], cache.strides[2]];

        self.prepare_tensors_for_active_cmd(&[&view])?;

        Ok((view, dims[1]))
    }

    pub(crate) fn sdpa_workspace_key_for(&self, tensor: &Tensor<T>) -> SdpaWorkspaceKey {
        SdpaWorkspaceKey::from_tensor(tensor)
    }

    pub(crate) fn sdpa_seq_delta(&mut self, key: SdpaWorkspaceKey, descriptor: SdpaKey, seq_q: usize, seq_k: usize) -> usize {
        let entry = self
            .sdpa_workspaces
            .entry(key)
            .or_insert_with(|| SdpaWorkspaceState::new(descriptor.clone()));

        if entry.descriptor != descriptor {
            entry.reset(descriptor);
        }

        let delta = seq_k.saturating_sub(entry.last_seq_k);
        entry.last_seq_q = seq_q;
        entry.last_seq_k = seq_k;

        if delta == 0 { seq_k } else { delta }
    }

    #[inline]
    pub fn scaled_dot_product_attention(
        &mut self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        causal: bool,
    ) -> Result<Tensor<T>, MetalError> {
        self.scaled_dot_product_attention_with_offset(q, k, v, causal, 0)
    }

    #[inline]
    pub fn scaled_dot_product_attention_with_offset(
        &mut self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        causal: bool,
        query_offset: usize,
    ) -> Result<Tensor<T>, MetalError> {
        // Use the kernel system for SDPA
        self.call::<ScaledDotProductAttentionOptimizedOp>((q, k, v, causal, query_offset as u32))
    }

    fn ensure_active_cmd_buffer(&mut self) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer_internal(true)
    }

    #[cfg(test)]
    pub(crate) fn force_enable_profiling_for_tests(&mut self) {
        crate::profiling_state::set_profiling_state(true);
    }

    fn ensure_active_cmd_buffer_internal(&mut self, ensure_cache: bool) -> Result<(), MetalError> {
        let should_refresh = if let Some(active) = self.active_cmd_buffer.as_ref() {
            if active.is_committed() {
                if !active.is_completed() {
                    active.wait();
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if should_refresh {
            self.active_cmd_buffer = None;
        }

        if self.active_cmd_buffer.is_none() {
            let cmd_buf = CommandBuffer::new(&self.command_queue)?;
            if let Some(profiler) = GpuProfiler::attach(&cmd_buf, crate::profiling_state::get_profiling_state()) {
                cmd_buf.retain_profiler(profiler);
            }
            self.active_cmd_buffer = Some(cmd_buf);
        }

        if ensure_cache && self.active_resource_cache.is_none() {
            self.active_resource_cache = Some(ResourceCache::with_device(self.device.clone()));
        }

        Ok(())
    }

    pub(crate) fn active_command_buffer_mut(&mut self) -> Result<&mut CommandBuffer, MetalError> {
        self.ensure_active_cmd_buffer()?;
        Ok(self.active_cmd_buffer.as_mut().expect("active command buffer must exist"))
    }

    pub(crate) fn active_command_buffer_mut_without_cache(&mut self) -> Result<&mut CommandBuffer, MetalError> {
        self.ensure_active_cmd_buffer_internal(false)?;
        Ok(self.active_cmd_buffer.as_mut().expect("active command buffer must exist"))
    }

    #[cfg(test)]
    pub(crate) fn materialize_contiguous_view(&mut self, view: Tensor<T>) -> Result<Tensor<T>, MetalError> {
        use crate::{TensorInit, TensorStorage};

        if view.strides == Tensor::<T>::compute_strides(view.dims()) {
            return Ok(view);
        }

        let dims = view.dims().to_vec();
        let contiguous = Tensor::new(dims, TensorStorage::Pooled(self), TensorInit::Uninitialized)?;

        self.prepare_tensors_for_active_cmd(&[&view])?;

        let source_view = view.as_mps_matrix_batch_view()?;
        let dest_view = contiguous.as_mps_matrix_batch_view()?;
        let elem_size = view.dtype.size_bytes();

        let command_buffer = self.active_command_buffer_mut_without_cache()?;
        let encoder = command_buffer.get_blit_encoder()?;

        for batch_idx in 0..source_view.batch {
            for row_idx in 0..source_view.rows {
                let src_offset = view.offset + batch_idx * source_view.matrix_bytes + row_idx * source_view.row_bytes;
                let dst_offset = contiguous.offset + batch_idx * dest_view.matrix_bytes + row_idx * dest_view.row_bytes;
                let copy_bytes = dest_view.columns * elem_size;
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &view.buf,
                        src_offset,
                        &contiguous.buf,
                        dst_offset,
                        copy_bytes,
                    );
                }
            }
        }

        self.mark_tensor_pending(&contiguous);
        self.finalize_active_command_buffer_if_latency();

        Ok(contiguous)
    }

    pub(crate) fn mark_tensor_pending(&self, tensor: &Tensor<T>) {
        tensor.mark_device_dirty();
        if let Some(active) = &self.active_cmd_buffer {
            tensor.defining_cmd_buffer.borrow_mut().replace(active.clone());
        }
    }

    fn prepare_tensor_for_active_cmd(&mut self, tensor: &Tensor<T>) -> Result<(), MetalError> {
        tensor.flush_host_writes()?;
        let maybe_dep = tensor.defining_cmd_buffer.borrow().clone();
        if let Some(dep) = maybe_dep {
            if self.active_cmd_buffer.as_ref().map(|active| dep.ptr_eq(active)).unwrap_or(false) {
                return Ok(());
            }

            if dep.is_completed() {
                tensor.defining_cmd_buffer.borrow_mut().take();
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
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: format!("{}/dep_wait", path),
                        backend: label.backend,
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                } else {
                    record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                        op_name: "Generation Loop/dep_wait".to_string(),
                        backend: GPU_PROFILER_BACKEND.to_string(),
                        duration_us: (waited.as_secs_f64() * 1e6).round() as u64,
                    });
                }
            }

            tensor.defining_cmd_buffer.borrow_mut().take();
        }
        Ok(())
    }

    pub(crate) fn prepare_tensors_for_active_cmd(&mut self, tensors: &[&Tensor<T>]) -> Result<(), MetalError> {
        for tensor in tensors {
            self.prepare_tensor_for_active_cmd(tensor)?;
        }
        Ok(())
    }
}
