use super::error::MetalError;
use super::instrumentation::{
    LatencyCollectorHandle, LatencyEvent, MatMulDispatchHandle, MatMulDispatchKind, MatMulDispatchRegistration, MatMulInstrumentation,
    MatMulSampleRecorder, MatmulDims, MemoryCollectorHandle, MemoryEvent, MemorySample, MemorySampleSender, MemoryUsage,
};
use super::operation::CommandBuffer;
use super::pool::MemoryPool;
use super::resource_cache::{CacheStats, ResourceCache};
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddInplaceOp;
use crate::metallic::kernels::swiglu::SwiGLUOp;
use crate::metallic::tensor::Dtype;
use crate::metallic::{Tensor, TensorElement, cache_keys::SdpaKey, kernels};
use kernels::gemv::GemvOp;
use kernels::kv_cache_write::{KvCacheWriteConfig, KvCacheWriteOp};
use kernels::matmul::{MatMulAlphaBetaOp, MatMulBackend, MatMulOp, MatMulSample};
use kernels::mlxmatmul::{MatMulMlxOp, MlxKernelCache};
use kernels::scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp;
use kernels::{KernelInvocable, KernelManager};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder as _;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const FORCE_MATMUL_BACKEND_ENV: &str = "FORCE_MATMUL_BACKEND";
const MATMUL_TRACE_ENV: &str = "METALLIC_LOG_MATMUL_SHAPES";
const MATMUL_TRACE_FILE_ENV: &str = "METALLIC_LOG_MATMUL_SHAPES_FILE";
const MATMUL_TRACE_DEFAULT_FILE: &str = "metal-matmul-shapes.log";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMulBackendOverride {
    Default,
    Force(MatMulBackend),
    Auto,
}

fn detect_forced_matmul_backend() -> MatMulBackendOverride {
    match env::var(FORCE_MATMUL_BACKEND_ENV) {
        Ok(value) => {
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
        Err(env::VarError::NotPresent) => MatMulBackendOverride::Default,
        Err(env::VarError::NotUnicode(_)) => MatMulBackendOverride::Default,
    }
}

fn env_flag_enabled(value: Result<String, env::VarError>) -> bool {
    match value {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                false
            } else {
                !matches!(trimmed.to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no")
            }
        }
        Err(env::VarError::NotPresent) => false,
        Err(env::VarError::NotUnicode(_)) => false,
    }
}

fn matmul_log_path_from_env() -> Option<PathBuf> {
    match env::var(MATMUL_TRACE_FILE_ENV) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Some(PathBuf::from(MATMUL_TRACE_DEFAULT_FILE))
            } else {
                Some(PathBuf::from(trimmed))
            }
        }
        Err(env::VarError::NotPresent) => Some(PathBuf::from(MATMUL_TRACE_DEFAULT_FILE)),
        Err(env::VarError::NotUnicode(_)) => None,
    }
}

fn matmul_shape_logger() -> Option<&'static MatmulShapeLogger> {
    static LOGGER: OnceLock<Option<MatmulShapeLogger>> = OnceLock::new();

    LOGGER
        .get_or_init(|| {
            matmul_log_path_from_env().and_then(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok()
                    .map(|file| MatmulShapeLogger { file: Mutex::new(file) })
            })
        })
        .as_ref()
}

#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct MemoryUsageCache {
    pool_used: usize,
    pool_capacity: usize,
    kv_used: usize,
    kv_capacity: usize,
    kv_cache_bytes: usize,
}

impl MemoryUsageCache {
    fn refresh(&mut self, pool: &MemoryPool, kv_pool: &MemoryPool, kv_cache_bytes: usize) -> MemoryUsage {
        self.pool_used = pool.used_bytes();
        self.pool_capacity = pool.total_capacity();
        self.kv_used = kv_pool.used_bytes();
        self.kv_capacity = kv_pool.total_capacity();
        self.kv_cache_bytes = kv_cache_bytes;
        self.current_usage()
    }

    fn current_usage(&self) -> MemoryUsage {
        MemoryUsage {
            pool_used: self.pool_used,
            pool_capacity: self.pool_capacity,
            kv_used: self.kv_used,
            kv_capacity: self.kv_capacity,
            kv_cache_bytes: self.kv_cache_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheWritePathStats {
    pub kernel_dispatches: usize,
    pub fallback_blits: usize,
}

impl KvCacheWritePathStats {
    pub fn total(&self) -> usize {
        self.kernel_dispatches + self.fallback_blits
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheDispatchStats {
    pub single_layout: KvCacheWritePathStats,
    pub fused_layout: KvCacheWritePathStats,
}

struct KvWritePlan<T: TensorElement> {
    k_src: Tensor<T>,
    v_src: Tensor<T>,
    k_cache: Tensor<T>,
    v_cache: Tensor<T>,
    canonical_heads: usize,
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

struct MatmulShapeLogger {
    file: Mutex<std::fs::File>,
}

impl MatmulShapeLogger {
    fn log_line(&self, line: &str) {
        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(file, "{line}");
        }
    }
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
    /// Optional latency collector used to report per-iteration timings.
    latency_collector: Option<LatencyCollectorHandle>,
    /// Optional memory collector used to capture detailed allocation snapshots.
    memory_collector: Option<MemoryCollectorHandle>,
    memory_sample_tx: Option<MemorySampleSender>,
    /// Shared instrumentation used to collect matmul GPU timings.
    matmul_instrumentation: MatMulInstrumentation,
    /// Matmul timing samples captured since the last drain.
    matmul_samples: Arc<Mutex<Vec<MatMulSample>>>,
    matmul_recorder: MatMulSampleRecorder,
    matmul_logging_session: RefCell<Option<Vec<MatMulDispatchHandle>>>,
    /// Workspace reused across sampling invocations to avoid per-token allocations.
    pub sampler_buffers: SamplerBuffers,
    /// Optional override for the matmul backend chosen by this context.
    forced_matmul_backend: MatMulBackendOverride,
    /// Whether to emit shape/timing logs for matmul calls.
    log_matmul_shapes: bool,
    /// Cache of MLX compute pipelines keyed by kernel configuration.
    pub(crate) mlx_kernel_cache: MlxKernelCache,
    /// Instrumentation for KV cache write paths.
    kv_cache_dispatch_stats: KvCacheDispatchStats,
    memory_usage_cache: MemoryUsageCache,
    sdpa_workspaces: FxHashMap<SdpaWorkspaceKey, SdpaWorkspaceState>,
    //config: ContextConfig,
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
    pub canonical_heads: usize,
    pub group_size: usize,
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
            cmd_buf.commit();
            cmd_buf.wait();
            return;
        }

        if let Some(cb) = self.command_queue.commandBuffer() {
            cb.commit();
            unsafe { cb.waitUntilCompleted() };
        }
    }

    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device, &command_queue)?;
        let kv_cache_pool = MemoryPool::with_limit(&device, &command_queue, KV_CACHE_POOL_MAX_BYTES)?;
        let forced_backend = detect_forced_matmul_backend();
        let log_matmul_shapes = env_flag_enabled(env::var(MATMUL_TRACE_ENV));

        if log_matmul_shapes {
            let _ = matmul_shape_logger();
        }

        let matmul_samples = Arc::new(Mutex::new(Vec::new()));
        let samples_for_recorder = Arc::clone(&matmul_samples);
        let matmul_recorder = MatMulSampleRecorder::new(move |sample| {
            if sample.duration.is_zero() {
                return;
            }
            if let Ok(mut samples) = samples_for_recorder.lock() {
                samples.push(sample);
            }
        });
        let matmul_instrumentation = MatMulInstrumentation::new(Some(&device));

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
            latency_collector: None,
            memory_collector: None,
            memory_sample_tx: None,
            matmul_instrumentation,
            matmul_samples,
            matmul_recorder,
            matmul_logging_session: RefCell::new(None),
            sampler_buffers: SamplerBuffers::default(),
            forced_matmul_backend: forced_backend,
            log_matmul_shapes,
            mlx_kernel_cache: MlxKernelCache::default(),
            kv_cache_dispatch_stats: KvCacheDispatchStats::default(),
            memory_usage_cache: MemoryUsageCache::default(),
            sdpa_workspaces: FxHashMap::default(),
            //config,
        })
    }

    pub fn tensor_dtype(&self) -> Dtype {
        T::DTYPE
    }

    pub fn call<K: KernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
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

        let command_buffer = self.active_cmd_buffer.as_mut().expect("active command buffer must exist");

        command_buffer.record(&*operation, &mut cache)?;

        self.active_resource_cache = Some(cache);

        self.mark_tensor_pending(&output);

        Ok(output)
    }

    /// Registers a latency collector handle for the upcoming operations. Passing `None`
    /// disables instrumentation and avoids the associated overhead.
    pub fn set_latency_collector(&mut self, collector: Option<LatencyCollectorHandle>) {
        self.latency_collector = collector;
    }

    /// Emit a latency event to the currently installed collector, if any.
    pub fn record_latency_event(&mut self, event: LatencyEvent<'_>, duration: Duration) {
        if let Some(collector) = self.latency_collector.as_ref() {
            collector.borrow_mut().record(event, duration);
        }
    }

    pub(crate) fn register_matmul_dispatch(
        &self,
        command_buffer: &CommandBuffer,
        backend: MatMulBackend,
        dims: Option<MatmulDims>,
        kind: MatMulDispatchKind,
    ) -> MatMulDispatchRegistration {
        let registration = self
            .matmul_instrumentation
            .register(command_buffer, backend, dims, kind, self.matmul_recorder.clone());
        self.track_matmul_dispatch(registration.handle());
        registration
    }

    fn track_matmul_dispatch(&self, handle: MatMulDispatchHandle) {
        let mut guard = self.matmul_logging_session.borrow_mut();
        if let Some(session) = guard.as_mut() {
            session.push(handle);
        }
    }

    fn begin_matmul_logging_session(&self) {
        *self.matmul_logging_session.borrow_mut() = Some(Vec::new());
    }

    fn finish_matmul_logging_session(&self) -> Vec<MatMulDispatchHandle> {
        self.matmul_logging_session.borrow_mut().take().unwrap_or_default()
    }

    #[allow(dead_code)]
    pub(crate) fn record_matmul_backend_sample(&self, backend: MatMulBackend, duration: Duration) {
        self.matmul_recorder.record(MatMulSample {
            backend,
            duration,
            dims: None,
            handle: None,
        });
    }

    pub fn take_matmul_samples(&self) -> Vec<MatMulSample> {
        let mut samples = match self.matmul_samples.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };
        samples.drain(..).collect()
    }

    fn matched_matmul_samples(&self, handles: &[MatMulDispatchHandle]) -> Vec<MatMulSample> {
        if handles.is_empty() {
            return Vec::new();
        }

        let guard = match self.matmul_samples.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };

        let mut matches = Vec::new();
        for sample in guard.iter() {
            if let Some(handle) = sample.handle
                && handles.contains(&handle)
            {
                matches.push(*sample);
            }
        }

        matches
    }

    pub fn kv_cache_dispatch_stats(&self) -> KvCacheDispatchStats {
        self.kv_cache_dispatch_stats
    }

    pub fn reset_kv_cache_dispatch_stats(&mut self) {
        self.kv_cache_dispatch_stats = KvCacheDispatchStats::default();
    }

    fn compute_matmul_dims(&self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<MatmulDims, MetalError> {
        let left_view = a.as_mps_matrix_batch_view()?;
        let right_view = b.as_mps_matrix_batch_view()?;

        if left_view.batch != right_view.batch {
            return Err(MetalError::InvalidOperation(
                "Left and right operands must share the same batch".to_string(),
            ));
        }

        let (a_rows, a_cols) = if transpose_a {
            (left_view.columns, left_view.rows)
        } else {
            (left_view.rows, left_view.columns)
        };
        let (b_rows, b_cols) = if transpose_b {
            (right_view.columns, right_view.rows)
        } else {
            (right_view.rows, right_view.columns)
        };

        if a_cols != b_rows {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices with shapes {}x{} and {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        Ok(MatmulDims {
            batch: left_view.batch,
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

    fn log_matmul_event(
        &self,
        op: &str,
        backend: MatMulBackend,
        dims: Option<&MatmulDims>,
        elapsed: Duration,
        note: &str,
        gpu_duration: Option<Duration>,
    ) {
        if !self.log_matmul_shapes {
            return;
        }

        let dispatch_ms = elapsed.as_secs_f64() * 1e3;
        let gpu_ms = gpu_duration.map(|d| d.as_secs_f64() * 1e3);
        let line = match dims {
            Some(d) => match gpu_ms {
                Some(g) => format!(
                    "[matmul-log] op={op} backend={backend:?} batch={} m={} n={} k={} dispatch_ms={dispatch_ms:.3} gpu_ms={g:.3} note={note}",
                    d.batch, d.m, d.n, d.k
                ),
                None => format!(
                    "[matmul-log] op={op} backend={backend:?} batch={} m={} n={} k={} dispatch_ms={dispatch_ms:.3} note={note}",
                    d.batch, d.m, d.n, d.k
                ),
            },
            None => match gpu_ms {
                Some(g) => format!("[matmul-log] op={op} backend={backend:?} dispatch_ms={dispatch_ms:.3} gpu_ms={g:.3} note={note}"),
                None => format!("[matmul-log] op={op} backend={backend:?} dispatch_ms={dispatch_ms:.3} note={note}"),
            },
        };

        if let Some(logger) = matmul_shape_logger() {
            logger.log_line(&line);
        }
    }

    fn with_matmul_logging<F>(
        &mut self,
        op: &str,
        backend: MatMulBackend,
        dims: Option<MatmulDims>,
        note: &str,
        f: F,
    ) -> Result<Tensor<T>, MetalError>
    where
        F: FnOnce(&mut Self) -> Result<Tensor<T>, MetalError>,
    {
        if self.log_matmul_shapes {
            self.begin_matmul_logging_session();

            let start = Instant::now();
            let result = f(self);
            let elapsed = start.elapsed();

            let handles = self.finish_matmul_logging_session();
            let gpu_duration = if result.is_ok() {
                let samples = self.matched_matmul_samples(&handles);
                let total: Duration = samples
                    .iter()
                    .filter(|sample| sample.backend == backend)
                    .map(|sample| sample.duration)
                    .sum();
                if total.is_zero() { None } else { Some(total) }
            } else {
                None
            };

            self.log_matmul_event(op, backend, dims.as_ref(), elapsed, note, gpu_duration);
            result
        } else {
            f(self)
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn matmul_bias_add_gemv_path(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        bias: &Tensor<T>,
        dims: Option<MatmulDims>,
        note: &str,
    ) -> Result<Tensor<T>, MetalError> {
        self.with_matmul_logging("matmul_bias_add", MatMulBackend::Gemv, dims, note, |ctx| {
            let mut linear = ctx.call::<GemvOp>((a, b))?;
            linear = ctx.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
            Ok(linear)
        })
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn matmul_bias_add_mlx_path(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        bias: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        dims: Option<MatmulDims>,
        note: &str,
    ) -> Result<Tensor<T>, MetalError> {
        self.with_matmul_logging("matmul_bias_add", MatMulBackend::Mlx, dims, note, |ctx| {
            ctx.call::<MatMulMlxOp>((a, b, Some(bias), None, transpose_a, transpose_b, 1.0, 0.0))
        })
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn matmul_bias_add_mps_path(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        bias: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        dims: Option<MatmulDims>,
        note: &str,
    ) -> Result<Tensor<T>, MetalError> {
        self.with_matmul_logging("matmul_bias_add", MatMulBackend::Mps, dims, note, |ctx| {
            let mut linear = ctx.call::<MatMulOp>((a, b, transpose_a, transpose_b))?;
            linear = ctx.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
            Ok(linear)
        })
    }

    /// Registers a memory collector handle for the upcoming operations. Passing `None`
    /// disables memory instrumentation.
    #[inline]
    pub fn set_memory_collector(&mut self, collector: Option<MemoryCollectorHandle>) {
        let sample_tx = collector.as_ref().map(|handle| handle.borrow().sample_sender());
        self.memory_sample_tx = sample_tx;
        self.memory_collector = collector;
    }

    /// Emit a memory event to the currently installed collector, capturing the latest
    /// allocation snapshot inside the callback.
    #[inline]
    pub fn record_memory_event(&mut self, event: MemoryEvent<'_>) {
        if let Some(sender) = self.memory_sample_tx.as_ref().cloned() {
            let usage = self.refresh_memory_usage();
            let sample = MemorySample {
                event: event.into_owned(),
                usage,
            };
            let _ = sender.send(sample);
        }
    }

    /// Capture a snapshot of the current memory usage for both the transient tensor pool
    /// and the persistent KV cache pool.
    #[inline]
    pub fn snapshot_memory_usage(&mut self) -> MemoryUsage {
        self.refresh_memory_usage()
    }

    #[inline]
    fn refresh_memory_usage(&mut self) -> MemoryUsage {
        self.memory_usage_cache
            .refresh(&self.pool, &self.kv_cache_pool, self.kv_cache_total_bytes)
    }

    #[inline]
    pub fn matmul(&mut self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            MatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul", MatMulBackend::Mlx, dims, "mode=forced-mlx", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                    })
                }
                MatMulBackend::Mps => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul", MatMulBackend::Mps, dims, "mode=forced-mps", |ctx| {
                        ctx.call::<MatMulOp>((a, b, transpose_a, transpose_b))
                    })
                }
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            let dims = if self.log_matmul_shapes { Some(dimensions) } else { None };
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.with_matmul_logging("matmul", MatMulBackend::Gemv, dims, "mode=forced-gemv", |ctx| {
                                    ctx.call::<GemvOp>((a, b))
                                })
                            } else {
                                self.with_matmul_logging("matmul", MatMulBackend::Mlx, dims, "mode=forced-gemv-fallback", |ctx| {
                                    ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                                })
                            }
                        }
                        Err(_) => self.with_matmul_logging("matmul", MatMulBackend::Mlx, None, "mode=forced-gemv-fallback", |ctx| {
                            ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                        }),
                    }
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.with_matmul_logging("matmul", MatMulBackend::Gemv, Some(dimensions), "mode=auto-gemv", |ctx| {
                                ctx.call::<GemvOp>((a, b))
                            });
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        let dims = Some(dimensions);

                        if use_mlx {
                            self.with_matmul_logging("matmul", MatMulBackend::Mlx, dims, "mode=auto", |ctx| {
                                ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            })
                        } else {
                            self.with_matmul_logging("matmul", MatMulBackend::Mps, dims, "mode=auto-fallback", |ctx| {
                                ctx.call::<MatMulOp>((a, b, transpose_a, transpose_b))
                            })
                        }
                    }
                    Err(_) => self.with_matmul_logging("matmul", MatMulBackend::Mlx, None, "mode=auto", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                    }),
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
                MatMulBackend::Mlx => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mlx, dims, "mode=forced-mlx", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                    })
                }
                MatMulBackend::Mps => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mps, dims, "mode=forced-mps", |ctx| {
                        Self::call_with_cache::<MatMulOp>(ctx, (a, b, transpose_a, transpose_b), cache)
                    })
                }
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            let dims = if self.log_matmul_shapes { Some(dimensions) } else { None };
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.with_matmul_logging("matmul_with_cache", MatMulBackend::Gemv, dims, "mode=forced-gemv", |ctx| {
                                    Self::call_with_cache::<GemvOp>(ctx, (a, b), cache)
                                })
                            } else {
                                self.with_matmul_logging(
                                    "matmul_with_cache",
                                    MatMulBackend::Mlx,
                                    dims,
                                    "mode=forced-gemv-fallback",
                                    |ctx| ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                                )
                            }
                        }
                        Err(_) => {
                            self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mlx, None, "mode=forced-gemv-fallback", |ctx| {
                                ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            })
                        }
                    }
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.with_matmul_logging(
                                "matmul_with_cache",
                                MatMulBackend::Gemv,
                                Some(dimensions),
                                "mode=auto-gemv",
                                |ctx| Self::call_with_cache::<GemvOp>(ctx, (a, b), cache),
                            );
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        let dims = Some(dimensions);

                        if use_mlx {
                            self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mlx, dims, "mode=auto", |ctx| {
                                ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            })
                        } else {
                            self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mps, dims, "mode=auto-fallback", |ctx| {
                                Self::call_with_cache::<MatMulOp>(ctx, (a, b, transpose_a, transpose_b), cache)
                            })
                        }
                    }
                    Err(_) => self.with_matmul_logging("matmul_with_cache", MatMulBackend::Mlx, None, "mode=auto", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                    }),
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
                MatMulBackend::Mlx => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.matmul_bias_add_mlx_path(a, b, bias, transpose_a, transpose_b, dims, "mode=forced-mlx")
                }
                MatMulBackend::Mps => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.matmul_bias_add_mps_path(a, b, bias, transpose_a, transpose_b, dims, "mode=forced-mps")
                }
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            let dims = if self.log_matmul_shapes { Some(dimensions) } else { None };
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.matmul_bias_add_gemv_path(a, b, bias, dims, "mode=forced-gemv")
                            } else {
                                self.matmul_bias_add_mlx_path(a, b, bias, transpose_a, transpose_b, dims, "mode=forced-gemv-fallback")
                            }
                        }
                        Err(_) => self.matmul_bias_add_mlx_path(a, b, bias, transpose_a, transpose_b, None, "mode=forced-gemv-fallback"),
                    }
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        // DEBT: GEMV Needs to be optimized so it can push past MPS.
                        // GEMV is still slightly slower than MPS
                        //if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                        //    return self.matmul_bias_add_gemv_path(a, b, bias, Some(dimensions), "mode=auto-gemv");
                        //}

                        // We've seen MPS and MLX both perform well at times here, since we're putting in more MLX improvements
                        //we might as well use it and try for better MLX improvements since they are so close right now
                        let use_mlx = self.should_use_mlx_bias(&dimensions);
                        let dims = Some(dimensions);

                        if use_mlx {
                            self.matmul_bias_add_mlx_path(a, b, bias, transpose_a, transpose_b, dims, "mode=auto")
                        } else {
                            self.matmul_bias_add_mps_path(a, b, bias, transpose_a, transpose_b, dims, "mode=auto-fallback")
                        }
                    }
                    Err(_) => self.matmul_bias_add_mps_path(a, b, bias, transpose_a, transpose_b, None, "mode=auto-fallback"),
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
                MatMulBackend::Mlx => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_alpha_beta", MatMulBackend::Mlx, dims, "mode=forced-mlx", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                    })
                }
                MatMulBackend::Mps => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_alpha_beta", MatMulBackend::Mps, dims, "mode=forced-mps", |ctx| {
                        ctx.call::<MatMulAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta))
                    })
                }
                MatMulBackend::Gemv => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_alpha_beta", MatMulBackend::Mlx, dims, "mode=forced-gemv-fallback", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                    })
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (dims, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.with_matmul_logging("matmul_alpha_beta", MatMulBackend::Mlx, dims, "mode=auto", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                    })
                } else {
                    self.with_matmul_logging("matmul_alpha_beta", MatMulBackend::Mps, dims, "mode=auto-fallback", |ctx| {
                        ctx.call::<MatMulAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta))
                    })
                }
            }
        }
    }

    pub(crate) fn call_with_cache<K: KernelInvocable>(
        &mut self,
        args: K::Args<'_, T>,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        self.ensure_active_cmd_buffer_internal(false)?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            Some(self.kernel_manager.get_pipeline(kernel_func, T::DTYPE, &self.device)?)
        } else {
            None
        };

        let (operation, output) = K::new(self, args, pipeline, Some(cache))?;

        let command_buffer = self.active_command_buffer_mut_without_cache()?;
        command_buffer.record(&*operation, cache)?;
        self.mark_tensor_pending(&output);

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
                MatMulBackend::Mlx => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_alpha_beta_with_cache", MatMulBackend::Mlx, dims, "mode=forced-mlx", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                    })
                }
                MatMulBackend::Mps => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging("matmul_alpha_beta_with_cache", MatMulBackend::Mps, dims, "mode=forced-mps", |ctx| {
                        Self::call_with_cache::<MatMulAlphaBetaOp>(ctx, (a, b, result, transpose_a, transpose_b, alpha, beta), cache)
                    })
                }
                MatMulBackend::Gemv => {
                    let dims = if self.log_matmul_shapes {
                        self.compute_matmul_dims(a, b, transpose_a, transpose_b).ok()
                    } else {
                        None
                    };
                    self.with_matmul_logging(
                        "matmul_alpha_beta_with_cache",
                        MatMulBackend::Mlx,
                        dims,
                        "mode=forced-gemv-fallback",
                        |ctx| ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
                    )
                }
            },
            MatMulBackendOverride::Default | MatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (dims, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.with_matmul_logging("matmul_alpha_beta_with_cache", MatMulBackend::Mlx, dims, "mode=auto", |ctx| {
                        ctx.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                    })
                } else {
                    self.with_matmul_logging(
                        "matmul_alpha_beta_with_cache",
                        MatMulBackend::Mps,
                        dims,
                        "mode=auto-fallback",
                        |ctx| Self::call_with_cache::<MatMulAlphaBetaOp>(ctx, (a, b, result, transpose_a, transpose_b, alpha, beta), cache),
                    )
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
        self.memory_usage_cache.kv_cache_bytes = 0;
        self.sdpa_workspaces.clear();
    }

    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [batch * n_kv_heads, seq_len, head_dim] storing only canonical heads.
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_kv_cache(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        canonical_batch_heads: usize,
        head_dim: usize,
        group_size: usize,
    ) -> Result<(), MetalError> {
        if group_size == 0 {
            return Err(MetalError::InvalidOperation("alloc_kv_cache requires a non-zero group size".into()));
        }

        let canonical_dims = vec![canonical_batch_heads, seq_len, head_dim];

        // Allocate K and V tensors directly from the dedicated KV cache pool
        let k_allocation = self.kv_cache_pool.alloc_tensor::<T>(canonical_dims.clone())?;
        let v_allocation = self.kv_cache_pool.alloc_tensor::<T>(canonical_dims)?;

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
        if let Some(encoder) = cmd_buf.raw().blitCommandEncoder() {
            encoder.fillBuffer_range_value(&k.buf, (k.offset..k.offset + k_size).into(), 0);
            encoder.fillBuffer_range_value(&v.buf, (v.offset..v.offset + v_size).into(), 0);
            encoder.endEncoding();
        } else {
            return Err(MetalError::OperationNotSupported("Blit encoder not available".into()));
        }

        self.mark_tensor_pending(&k);
        self.mark_tensor_pending(&v);

        let entry = KvCacheEntry {
            k,
            v,
            dtype,
            element_size,
            zeroing_complete: true,
            capacity: seq_len,
            canonical_heads: canonical_batch_heads,
            group_size,
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
        self.memory_usage_cache.kv_cache_bytes = self.kv_cache_total_bytes;
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
        let expected_group_size = entry.group_size;
        let expected_canonical_heads = entry.canonical_heads;

        if group_size == 0 {
            return Err(MetalError::InvalidOperation("write_kv_step requires a non-zero group size".into()));
        }

        if group_size != expected_group_size {
            return Err(MetalError::InvalidOperation(format!(
                "write_kv_step received group size {} but cache registered {} for layer {}",
                group_size, expected_group_size, layer_idx
            )));
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

        let expected_cache_heads = cache_dims[0];
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

        if bh != expected_canonical_heads {
            return Err(MetalError::DimensionMismatch {
                expected: expected_canonical_heads,
                actual: bh,
            });
        }

        if expected_cache_heads != expected_canonical_heads {
            return Err(MetalError::DimensionMismatch {
                expected: expected_canonical_heads,
                actual: expected_cache_heads,
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
        let heads_u32 =
            u32::try_from(expected_canonical_heads).map_err(|_| MetalError::InvalidShape("batch_heads exceeds u32::MAX".into()))?;
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
            canonical_heads: expected_canonical_heads,
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
            src_head_stride,
            src_seq_stride,
            dst_head_stride,
            dst_seq_stride,
            total_threads,
        };

        match self.call::<KvCacheWriteOp>((k_src.clone(), v_src.clone(), k_cache.clone(), v_cache.clone(), config)) {
            Ok(_) => {
                self.kv_cache_dispatch_stats.single_layout.kernel_dispatches += 1;
            }
            Err(err) if Self::kv_cache_kernel_unavailable(&err) => {
                self.kv_cache_dispatch_stats.single_layout.fallback_blits += 1;
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
                );
            }
            Err(err) => return Err(err),
        }

        if let Some(entry) = self.kv_caches.get_mut(&layer_idx) {
            entry.zeroing_complete = false;
        }

        self.mark_tensor_pending(&k_cache);
        self.mark_tensor_pending(&v_cache);

        Ok(())
    }
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
    ) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer()?;
        let encoder = {
            let cmd_buf = self.active_command_buffer_mut()?;
            cmd_buf
                .raw()
                .blitCommandEncoder()
                .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?
        };

        let cache_stride_elems = capacity * head_dim;
        let copy_bytes = head_dim * element_size;
        let k_dims_len = k_src.dims().len();
        let v_dims_len = v_src.dims().len();

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
                let dst_elem_index = head_idx * cache_stride_elems + step * head_dim;

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

        encoder.endEncoding();

        self.mark_tensor_pending(k_cache);
        self.mark_tensor_pending(v_cache);

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

    pub fn kv_repeat_view(
        &mut self,
        canonical: &Tensor<T>,
        group_size: usize,
        batch: usize,
        n_kv_heads: usize,
        n_heads: usize,
        active_seq: usize,
    ) -> Result<Tensor<T>, MetalError> {
        if group_size == 0 {
            return Err(MetalError::InvalidShape(
                "kv_repeat_view requires a non-zero group size".to_string(),
            ));
        }
        if n_kv_heads == 0 {
            return Err(MetalError::InvalidShape(
                "kv_repeat_view requires at least one KV head".to_string(),
            ));
        }
        if n_heads != n_kv_heads * group_size {
            return Err(MetalError::InvalidShape(format!(
                "kv_repeat_view expected n_heads {} to equal n_kv_heads {} * group_size {}",
                n_heads, n_kv_heads, group_size
            )));
        }

        let dims = canonical.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape(
                "kv_repeat_view expects canonical tensors shaped [batch*n_kv_heads, seq, head_dim]".to_string(),
            ));
        }

        let expected_heads = batch
            .checked_mul(n_kv_heads)
            .ok_or_else(|| MetalError::InvalidShape("kv_repeat_view head count overflow".to_string()))?;
        if dims[0] != expected_heads {
            return Err(MetalError::InvalidShape(format!(
                "kv_repeat_view expected {} canonical heads, found {}",
                expected_heads, dims[0]
            )));
        }
        if active_seq == 0 || active_seq > dims[1] {
            return Err(MetalError::InvalidShape(format!(
                "kv_repeat_view active sequence {} exceeds available {}",
                active_seq, dims[1]
            )));
        }

        let mut view = canonical.clone();
        view.dims = vec![expected_heads, active_seq, dims[2]];
        view.strides = canonical.strides.clone();

        self.prepare_tensors_for_active_cmd(&[&view])?;

        Ok(view)
    }

    pub(crate) fn sdpa_workspace_key_for(&self, tensor: &Tensor<T>) -> SdpaWorkspaceKey {
        SdpaWorkspaceKey::from_tensor(tensor)
    }

    pub(crate) fn reset_sdpa_workspace(&mut self, key: SdpaWorkspaceKey) {
        self.sdpa_workspaces.remove(&key);
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
        self.scaled_dot_product_attention_with_group(q, k, v, causal, 0, 1)
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
        self.scaled_dot_product_attention_with_group(q, k, v, causal, query_offset, 1)
    }

    #[inline]
    pub fn scaled_dot_product_attention_with_group(
        &mut self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        causal: bool,
        query_offset: usize,
        group_size: usize,
    ) -> Result<Tensor<T>, MetalError> {
        // Use the kernel system for SDPA
        self.call::<ScaledDotProductAttentionOptimizedOp>(
            (q, k, v, causal, query_offset as u32, group_size as u32),
        )
    }

    /// SwiGLU implementation extracted from Qwen25 FFN block.
    /// Computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    ///
    /// # Arguments
    /// * `x_normed_flat` - Flattened input [m, d_model] where m = batch * seq
    /// * `ffn_gate` - Gate projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
    /// * `ffn_up` - Up projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
    /// * `ffn_down` - Down projection weight [d_model, ff_dim] (row-major; transpose if source stored as [ff_dim, d_model])
    /// * `fused_gate_up_weight` - Optional fused gate/up weight storing both projections in a single matrix
    /// * `ctx` - Metal context for operations
    ///
    /// # Returns
    /// Flat output [m, d_model] (reshape externally to [batch, seq, d_model])
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn SwiGLU(
        &mut self,
        x_normed_flat: &Tensor<T>,
        ffn_gate: &Tensor<T>,
        ffn_gate_bias: &Tensor<T>,
        ffn_up: &Tensor<T>,
        ffn_up_bias: &Tensor<T>,
        ffn_down: &Tensor<T>,
        ffn_down_bias: &Tensor<T>,
        fused_gate_up_weight: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>, MetalError> {
        // Use the kernel system to call the SwiGLU operation
        self.call::<SwiGLUOp>((
            x_normed_flat,
            ffn_gate,
            ffn_gate_bias,
            ffn_up,
            ffn_up_bias,
            ffn_down,
            ffn_down_bias,
            fused_gate_up_weight,
        ))
    }

    fn ensure_active_cmd_buffer(&mut self) -> Result<(), MetalError> {
        self.ensure_active_cmd_buffer_internal(true)
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
        use crate::metallic::{TensorInit, TensorStorage};

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
        let encoder = command_buffer
            .raw()
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".to_string()))?;

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

        encoder.endEncoding();
        self.mark_tensor_pending(&contiguous);

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

            dep.commit();
            dep.wait();
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
