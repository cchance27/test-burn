use super::error::MetalError;
use super::instrumentation::{
    LatencyCollectorHandle, LatencyEvent, MatMulInstrumentation, MatMulSampleRecorder, MemoryCollectorHandle, MemoryEvent, MemoryUsage,
};
use super::operation::CommandBuffer;
use super::pool::MemoryPool;
use super::resource_cache::{CacheStats, ResourceCache};
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddInplaceOp;
use crate::metallic::kernels::fused_rmsnorm_qkv::FusedRmsNormQkvProjectionOp;
use crate::metallic::kernels::swiglu::SwiGLUOp;
use crate::metallic::tensor::Dtype;
use crate::metallic::{Tensor, TensorElement, TensorInit, TensorStorage, kernels};
use kernels::matmul::{MatMulAlphaBetaOp, MatMulBackend, MatMulOp, MatMulSample};
use kernels::scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp;
use kernels::{KernelInvocable, KernelManager};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder as _;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
pub struct SamplerBuffers {
    pub scaled: Vec<f32>,
    pub indices: Vec<usize>,
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

    /// Lazily created command buffer used to batch kernel dispatches until synchronization.
    active_cmd_buffer: Option<CommandBuffer>,
    /// Resource cache associated with the active command buffer.
    active_resource_cache: Option<ResourceCache>,
    /// Optional latency collector used to report per-iteration timings.
    latency_collector: Option<LatencyCollectorHandle>,
    /// Optional memory collector used to capture detailed allocation snapshots.
    memory_collector: Option<MemoryCollectorHandle>,
    /// Shared instrumentation used to collect matmul GPU timings.
    matmul_instrumentation: MatMulInstrumentation,
    /// Matmul timing samples captured since the last drain.
    matmul_samples: Arc<Mutex<Vec<MatMulSample>>>,
    matmul_recorder: MatMulSampleRecorder,
    /// Workspace reused across sampling invocations to avoid per-token allocations.
    pub sampler_buffers: SamplerBuffers,
    //config: ContextConfig,
}

#[derive(Clone)]
pub(crate) struct KvCacheEntry<T: TensorElement> {
    pub k: Tensor<T>,
    pub v: Tensor<T>,
    pub repeated_k: Tensor<T>,
    pub repeated_v: Tensor<T>,
    #[allow(dead_code)]
    pub dtype: Dtype,
    pub element_size: usize,
    pub zeroing_complete: bool,
    pub capacity: usize,
}

const KV_CACHE_POOL_MAX_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB

impl<T: TensorElement> Context<T> {
    /// Synchronize pending GPU work, committing and waiting on the active command buffer.
    /// Falls back to the legacy submit/wait path if no active buffer exists.
    pub fn synchronize(&mut self) {
        if let Some(cmd_buf) = self.active_cmd_buffer.take() {
            cmd_buf.commit();
            cmd_buf.wait();
            self.active_resource_cache = None;
            return;
        }

        if let Some(cb) = self.command_queue.commandBuffer() {
            cb.commit();
            unsafe { cb.waitUntilCompleted() };
        }

        self.active_resource_cache = None;
    }

    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device, &command_queue)?;
        let kv_cache_pool = MemoryPool::with_limit(&device, &command_queue, KV_CACHE_POOL_MAX_BYTES)?;

        let matmul_samples = Arc::new(Mutex::new(Vec::new()));
        let samples_for_recorder = Arc::clone(&matmul_samples);
        let matmul_recorder = MatMulSampleRecorder::new(move |backend, duration| {
            if duration.is_zero() {
                return;
            }
            if let Ok(mut samples) = samples_for_recorder.lock() {
                samples.push(MatMulSample { backend, duration });
            }
        });

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
            active_cmd_buffer: None,
            active_resource_cache: None,
            latency_collector: None,
            memory_collector: None,
            matmul_instrumentation: MatMulInstrumentation::default(),
            matmul_samples,
            matmul_recorder,
            sampler_buffers: SamplerBuffers::default(),
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

    pub(crate) fn register_matmul_dispatch(&self, command_buffer: &CommandBuffer, backend: MatMulBackend) {
        self.matmul_instrumentation
            .register(command_buffer, backend, self.matmul_recorder.clone());
    }

    #[allow(dead_code)]
    pub(crate) fn record_matmul_backend_sample(&self, backend: MatMulBackend, duration: Duration) {
        self.matmul_recorder.record_matmul_backend_sample(backend, duration);
    }

    pub fn take_matmul_samples(&self) -> Vec<MatMulSample> {
        let mut samples = match self.matmul_samples.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };
        samples.drain(..).collect()
    }

    /// Registers a memory collector handle for the upcoming operations. Passing `None`
    /// disables memory instrumentation.
    pub fn set_memory_collector(&mut self, collector: Option<MemoryCollectorHandle>) {
        self.memory_collector = collector;
    }

    /// Emit a memory event to the currently installed collector, capturing the latest
    /// allocation snapshot inside the callback.
    pub fn record_memory_event(&mut self, event: MemoryEvent<'_>) {
        if let Some(collector) = self.memory_collector.as_ref() {
            let usage = self.snapshot_memory_usage();
            collector.borrow_mut().record(event, usage);
        }
    }

    /// Capture a snapshot of the current memory usage for both the transient tensor pool
    /// and the persistent KV cache pool.
    pub fn snapshot_memory_usage(&self) -> MemoryUsage {
        let kv_cache_bytes = self
            .kv_caches
            .values()
            .map(|entry| entry.k.size_bytes() + entry.v.size_bytes() + entry.repeated_k.size_bytes() + entry.repeated_v.size_bytes())
            .sum();

        MemoryUsage {
            pool_used: self.pool.used_bytes(),
            pool_capacity: self.pool.total_capacity(),
            kv_used: self.kv_cache_pool.used_bytes(),
            kv_capacity: self.kv_cache_pool.total_capacity(),
            kv_cache_bytes,
        }
    }

    #[inline]
    pub fn matmul(&mut self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<Tensor<T>, MetalError> {
        // Use the kernel system for matmul
        self.call::<MatMulOp>((a, b, transpose_a, transpose_b))
    }

    pub(crate) fn matmul_with_cache(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        self.call_with_cache::<MatMulOp>((a, b, transpose_a, transpose_b), cache)
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

        let mut linear = self.matmul(x_flat, fused_weight, false, false)?;
        linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, fused_bias.clone()))?;

        let q_range_end = d_model;
        let k_range_end = d_model + kv_dim;
        let v_range_end = expected_total;

        let q_out = linear.slice_last_dim(0..q_range_end)?;
        let k_out = linear.slice_last_dim(d_model..k_range_end)?;
        let v_out = linear.slice_last_dim(k_range_end..v_range_end)?;

        Ok((q_out, k_out, v_out))
    }

    #[allow(clippy::type_complexity)]
    pub fn fused_rmsnorm_qkv_projection(
        &mut self,
        x_flat: &Tensor<T>,
        gamma: &Tensor<T>,
        fused_weight: &Tensor<T>,
        fused_bias: &Tensor<T>,
        d_model: usize,
        kv_dim: usize,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        let x_dims = x_flat.dims();
        if x_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "fused_rmsnorm_qkv_projection expects a 2D input [m, d_model], got {:?}",
                x_dims
            )));
        }

        let rows = x_dims[0];
        let in_features = x_dims[1];
        if in_features != d_model {
            return Err(MetalError::InvalidShape(format!(
                "Input feature size {} does not match d_model {}",
                in_features, d_model
            )));
        }

        if gamma.dims() != [d_model] {
            return Err(MetalError::InvalidShape(format!(
                "Gamma dims {:?} incompatible with d_model {}",
                gamma.dims(),
                d_model
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

        let total_out_dim = expected_total as u32;
        let feature_dim = d_model as u32;

        let combined = self.call::<FusedRmsNormQkvProjectionOp>((
            x_flat.clone(),
            gamma.clone(),
            fused_weight.clone(),
            fused_bias.clone(),
            feature_dim,
            total_out_dim,
        ))?;

        let q_range_end = d_model;
        let k_range_end = d_model + kv_dim;
        let v_range_end = expected_total;

        let q_out = combined.slice_last_dim(0..q_range_end)?;
        let k_out = combined.slice_last_dim(d_model..k_range_end)?;
        let v_out = combined.slice_last_dim(k_range_end..v_range_end)?;

        // Each row of the fused output stores Q, K, and V segments back-to-back, so the
        // leading stride must remain `total_out_dim`. Consumers like `KvRearrangeOp`
        // depend on that stride to skip over the interleaved sections without forcing a
        // materialized copy.
        Ok((q_out, k_out, v_out))
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
        // Use the kernel system for matmul with alpha/beta scaling
        self.call::<MatMulAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta))
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
        self.call_with_cache::<MatMulAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta), cache)
    }

    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.active_resource_cache.as_ref().map(|cache| cache.get_stats())
    }

    pub fn clear_cache(&mut self) {
        if let Some(cache) = self.active_resource_cache.as_mut() {
            cache.clear();
        }
    }

    pub fn reset_pool(&mut self) {
        self.pool.reset();
    }

    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: canonical [batch * n_kv_heads, seq_len, head_dim] and repeated [batch * n_heads, seq_len, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_kv_cache(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        canonical_batch_heads: usize,
        repeated_batch_heads: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        let canonical_dims = vec![canonical_batch_heads, seq_len, head_dim];
        let repeated_dims = vec![repeated_batch_heads, seq_len, head_dim];

        // Allocate K and V tensors directly from the dedicated KV cache pool
        let k_allocation = self.kv_cache_pool.alloc_tensor::<T>(canonical_dims.clone())?;
        let v_allocation = self.kv_cache_pool.alloc_tensor::<T>(canonical_dims)?;
        let repeated_k_allocation = self.kv_cache_pool.alloc_tensor::<T>(repeated_dims.clone())?;
        let repeated_v_allocation = self.kv_cache_pool.alloc_tensor::<T>(repeated_dims)?;

        let dtype = k_allocation.dtype();
        let element_size = k_allocation.element_size();
        debug_assert_eq!(dtype, v_allocation.dtype());
        debug_assert_eq!(dtype, repeated_k_allocation.dtype());
        debug_assert_eq!(dtype, repeated_v_allocation.dtype());

        let k = k_allocation.into_tensor();
        let v = v_allocation.into_tensor();
        let repeated_k = repeated_k_allocation.into_tensor();
        let repeated_v = repeated_v_allocation.into_tensor();

        // Manually zero the tensors using a blit command
        let k_size = k.size_bytes();
        let v_size = v.size_bytes();
        let repeated_k_size = repeated_k.size_bytes();
        let repeated_v_size = repeated_v.size_bytes();

        self.ensure_active_cmd_buffer()?;
        let cmd_buf = self.active_command_buffer_mut()?;
        if let Some(encoder) = cmd_buf.raw().blitCommandEncoder() {
            encoder.fillBuffer_range_value(&k.buf, (k.offset..k.offset + k_size).into(), 0);
            encoder.fillBuffer_range_value(&v.buf, (v.offset..v.offset + v_size).into(), 0);
            encoder.fillBuffer_range_value(&repeated_k.buf, (repeated_k.offset..repeated_k.offset + repeated_k_size).into(), 0);
            encoder.fillBuffer_range_value(&repeated_v.buf, (repeated_v.offset..repeated_v.offset + repeated_v_size).into(), 0);
            encoder.endEncoding();
        } else {
            return Err(MetalError::OperationNotSupported("Blit encoder not available".into()));
        }

        self.mark_tensor_pending(&k);
        self.mark_tensor_pending(&v);
        self.mark_tensor_pending(&repeated_k);
        self.mark_tensor_pending(&repeated_v);

        self.kv_caches.insert(
            layer_idx,
            KvCacheEntry {
                k,
                v,
                repeated_k,
                repeated_v,
                dtype,
                element_size,
                zeroing_complete: true,
                capacity: seq_len,
            },
        );
        Ok(())
    }

    /// Write a single timestep of K and V (per-head flattened) into the per-layer cache at index `step`.
    /// - `k_step` and `v_step` must be contiguous tensors with shape [batch_heads, head_dim] or [batch_heads, 1, head_dim].
    ///   This performs a device blit copy from the source buffer into the cache at the correct offset.
    pub fn write_kv_step(&mut self, layer_idx: usize, step: usize, k_step: &Tensor<T>, v_step: &Tensor<T>) -> Result<(), MetalError> {
        let zero_ready = self.kv_caches.get(&layer_idx).map(|entry| entry.zeroing_complete).unwrap_or(false);

        let k_src = k_step.clone();
        let v_src = v_step.clone();

        let dims = k_src.dims().to_vec();
        let (bh, seq_in_src, hd) = match dims.len() {
            2 => (dims[0], 1, dims[1]),
            3 => (dims[0], dims[1], dims[2]),
            _ => (0, 0, 0),
        };

        let v_dims = v_src.dims().to_vec();
        let (v_bh, v_seq_in_src, v_hd) = match v_dims.len() {
            2 => (v_dims[0], 1, v_dims[1]),
            3 => (v_dims[0], v_dims[1], v_dims[2]),
            _ => (0, 0, 0),
        };

        if zero_ready {
            self.prepare_tensors_for_active_cmd(&[&k_src, &v_src])?;
            self.ensure_active_cmd_buffer()?;
            let encoder = {
                let cmd_buf = self.active_command_buffer_mut()?;
                cmd_buf
                    .raw()
                    .blitCommandEncoder()
                    .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?
            };

            let active_cmd_clone = self.active_cmd_buffer.clone();

            {
                let entry = self
                    .kv_caches
                    .get_mut(&layer_idx)
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not allocated", layer_idx)))?;

                if step >= entry.capacity {
                    return Err(MetalError::InvalidOperation(format!(
                        "Step {} exceeds KV cache capacity {} for layer {}",
                        step, entry.capacity, layer_idx
                    )));
                }

                let expected_bh = entry.k.dims()[0];
                let expected_hd = entry.k.dims()[2];
                if bh != expected_bh || hd != expected_hd {
                    return Err(MetalError::DimensionMismatch {
                        expected: expected_bh * expected_hd,
                        actual: bh * hd,
                    });
                }

                if seq_in_src != 1 {
                    return Err(MetalError::OperationNotSupported(
                        "write_kv_step currently expects a single timestep in the source tensor".into(),
                    ));
                }

                if step + seq_in_src > entry.capacity {
                    return Err(MetalError::InvalidOperation(format!(
                        "Writing KV step {} ({} timesteps) exceeds cache capacity {} for layer {}",
                        step, seq_in_src, entry.capacity, layer_idx
                    )));
                }

                if v_bh != expected_bh || v_hd != expected_hd {
                    return Err(MetalError::DimensionMismatch {
                        expected: expected_bh * expected_hd,
                        actual: v_bh * v_hd,
                    });
                }
                if v_seq_in_src != seq_in_src {
                    return Err(MetalError::OperationNotSupported(
                        "write_kv_step expects matching sequence dims for K and V".into(),
                    ));
                }

                let row_elems = expected_hd;
                let copy_bytes = row_elems * entry.element_size;
                let cache_stride_elems = entry.capacity * expected_hd;

                unsafe {
                    for head_idx in 0..expected_bh {
                        let src_elem_index = match dims.len() {
                            2 => head_idx * expected_hd,
                            3 => (head_idx * seq_in_src) * expected_hd,
                            _ => unreachable!(),
                        };
                        let v_src_elem_index = match v_dims.len() {
                            2 => head_idx * expected_hd,
                            3 => (head_idx * v_seq_in_src) * expected_hd,
                            _ => unreachable!(),
                        };
                        let dst_elem_index = head_idx * cache_stride_elems + step * expected_hd;

                        let src_offset_k = k_src.offset + src_elem_index * entry.element_size;
                        let src_offset_v = v_src.offset + v_src_elem_index * entry.element_size;
                        let dst_offset_k = entry.k.offset + dst_elem_index * entry.element_size;
                        let dst_offset_v = entry.v.offset + dst_elem_index * entry.element_size;

                        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &k_src.buf,
                            src_offset_k,
                            &entry.k.buf,
                            dst_offset_k,
                            copy_bytes,
                        );
                        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &v_src.buf,
                            src_offset_v,
                            &entry.v.buf,
                            dst_offset_v,
                            copy_bytes,
                        );
                    }
                }

                entry.zeroing_complete = false;
                if let Some(active) = active_cmd_clone {
                    entry.k.defining_cmd_buffer.borrow_mut().replace(active.clone());
                    entry.k.mark_device_dirty();
                    entry.v.defining_cmd_buffer.borrow_mut().replace(active);
                    entry.v.mark_device_dirty();
                }
            }

            encoder.endEncoding();

            return Ok(());
        }

        // Lookup the canonical cache tensors and clone their handles for the general path.
        let (k_cache_ref, v_cache_ref, capacity_seq_val, element_size) = match self.kv_caches.get(&layer_idx) {
            Some(entry) => (entry.k.clone(), entry.v.clone(), entry.capacity, entry.element_size),
            None => {
                return Err(MetalError::InvalidOperation(format!(
                    "KV cache for layer {} not allocated",
                    layer_idx
                )));
            }
        };
        if step >= capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Step {} exceeds KV cache capacity {} for layer {}",
                step, capacity_seq_val, layer_idx
            )));
        }

        let k_cache = k_cache_ref;
        let v_cache = v_cache_ref;
        self.prepare_tensors_for_active_cmd(&[&k_cache, &v_cache, &k_src, &v_src])?;

        let expected_bh = k_cache.dims()[0];
        let expected_hd = k_cache.dims()[2];
        if bh != expected_bh || hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_bh * expected_hd,
                actual: bh * hd,
            });
        }

        if seq_in_src != 1 {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step currently expects a single timestep in the source tensor".into(),
            ));
        }

        if step + seq_in_src > capacity_seq_val {
            return Err(MetalError::InvalidOperation(format!(
                "Writing KV step {} ({} timesteps) exceeds cache capacity {} for layer {}",
                step, seq_in_src, capacity_seq_val, layer_idx
            )));
        }

        if v_bh != expected_bh || v_hd != expected_hd {
            return Err(MetalError::DimensionMismatch {
                expected: expected_bh * expected_hd,
                actual: v_bh * v_hd,
            });
        }
        if v_seq_in_src != seq_in_src {
            return Err(MetalError::OperationNotSupported(
                "write_kv_step expects matching sequence dims for K and V".into(),
            ));
        }

        let row_elems = expected_hd;
        let copy_bytes = row_elems * element_size;
        let cache_stride_elems = capacity_seq_val * expected_hd;

        self.ensure_active_cmd_buffer()?;
        {
            let encoder = {
                let cmd_buf = self.active_command_buffer_mut()?;
                cmd_buf
                    .raw()
                    .blitCommandEncoder()
                    .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?
            };

            unsafe {
                for head_idx in 0..expected_bh {
                    let src_elem_index = match dims.len() {
                        2 => head_idx * expected_hd,
                        3 => (head_idx * seq_in_src) * expected_hd,
                        _ => unreachable!(),
                    };
                    let v_src_elem_index = match v_dims.len() {
                        2 => head_idx * expected_hd,
                        3 => (head_idx * v_seq_in_src) * expected_hd,
                        _ => unreachable!(),
                    };
                    let dst_elem_index = head_idx * cache_stride_elems + step * expected_hd;

                    let src_offset_k = k_src.offset + src_elem_index * element_size;
                    let src_offset_v = v_src.offset + v_src_elem_index * element_size;
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
        }

        self.mark_tensor_pending(&k_cache);
        self.mark_tensor_pending(&v_cache);

        if let Some(entry) = self.kv_caches.get_mut(&layer_idx) {
            entry.zeroing_complete = false;
        }

        Ok(())
    }

    /// Append a single timestep into the repeated KV cache by copying the canonical slice.
    #[allow(clippy::too_many_arguments)]
    pub fn write_repeated_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        group_size: usize,
        k_step: &Tensor<T>,
        v_step: &Tensor<T>,
    ) -> Result<(), MetalError> {
        if group_size == 0 {
            return Err(MetalError::InvalidOperation(
                "write_repeated_kv_step requires a non-zero group size".into(),
            ));
        }

        let zero_ready = self.kv_caches.get(&layer_idx).map(|entry| entry.zeroing_complete).unwrap_or(false);

        let k_src = k_step.clone();
        let v_src = v_step.clone();

        let k_dims = k_src.dims().to_vec();
        let (canonical_heads, seq_in_src, head_dim) = match k_dims.len() {
            2 => (k_dims[0], 1, k_dims[1]),
            3 => (k_dims[0], k_dims[1], k_dims[2]),
            _ => (0, 0, 0),
        };
        if seq_in_src != 1 {
            return Err(MetalError::OperationNotSupported(
                "write_repeated_kv_step expects a single timestep in the source tensor".into(),
            ));
        }

        let v_dims = v_src.dims().to_vec();
        let (v_heads, v_seq_in_src, v_head_dim) = match v_dims.len() {
            2 => (v_dims[0], 1, v_dims[1]),
            3 => (v_dims[0], v_dims[1], v_dims[2]),
            _ => (0, 0, 0),
        };

        if zero_ready {
            self.prepare_tensors_for_active_cmd(&[&k_src, &v_src])?;
            self.ensure_active_cmd_buffer()?;
            let encoder = {
                let cmd_buf = self.active_command_buffer_mut()?;
                cmd_buf
                    .raw()
                    .blitCommandEncoder()
                    .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?
            };

            let active_cmd_clone = self.active_cmd_buffer.clone();

            {
                let entry = self
                    .kv_caches
                    .get_mut(&layer_idx)
                    .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not allocated", layer_idx)))?;

                if step >= entry.capacity {
                    return Err(MetalError::InvalidOperation(format!(
                        "Step {} exceeds KV cache capacity {} for layer {}",
                        step, entry.capacity, layer_idx
                    )));
                }

                let repeated_heads_expected = canonical_heads
                    .checked_mul(group_size)
                    .ok_or_else(|| MetalError::InvalidOperation("group_size overflow while expanding KV heads".into()))?;

                if entry.repeated_k.dims()[0] != repeated_heads_expected || entry.repeated_k.dims()[2] != head_dim {
                    return Err(MetalError::DimensionMismatch {
                        expected: repeated_heads_expected * head_dim,
                        actual: entry.repeated_k.dims()[0] * entry.repeated_k.dims()[2],
                    });
                }

                if v_heads != canonical_heads || v_head_dim != head_dim || v_seq_in_src != seq_in_src {
                    return Err(MetalError::DimensionMismatch {
                        expected: canonical_heads * head_dim,
                        actual: v_heads * v_head_dim,
                    });
                }

                let copy_bytes = head_dim * entry.element_size;
                let repeated_stride_elems = entry.capacity * head_dim;

                unsafe {
                    for kv_head in 0..canonical_heads {
                        let src_elem_index = match k_dims.len() {
                            2 => kv_head * head_dim,
                            3 => (kv_head * seq_in_src) * head_dim,
                            _ => unreachable!(),
                        };
                        let v_src_elem_index = match v_dims.len() {
                            2 => kv_head * head_dim,
                            3 => (kv_head * v_seq_in_src) * head_dim,
                            _ => unreachable!(),
                        };

                        let src_offset_k = k_src.offset + src_elem_index * entry.element_size;
                        let src_offset_v = v_src.offset + v_src_elem_index * entry.element_size;

                        for group in 0..group_size {
                            let repeated_head = kv_head * group_size + group;
                            let dst_elem_index = repeated_head * repeated_stride_elems + step * head_dim;

                            let dst_offset_k = entry.repeated_k.offset + dst_elem_index * entry.element_size;
                            let dst_offset_v = entry.repeated_v.offset + dst_elem_index * entry.element_size;

                            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &k_src.buf,
                                src_offset_k,
                                &entry.repeated_k.buf,
                                dst_offset_k,
                                copy_bytes,
                            );
                            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &v_src.buf,
                                src_offset_v,
                                &entry.repeated_v.buf,
                                dst_offset_v,
                                copy_bytes,
                            );
                        }
                    }
                }

                entry.zeroing_complete = false;
                if let Some(active) = active_cmd_clone {
                    entry.repeated_k.defining_cmd_buffer.borrow_mut().replace(active.clone());
                    entry.repeated_k.mark_device_dirty();
                    entry.repeated_v.defining_cmd_buffer.borrow_mut().replace(active);
                    entry.repeated_v.mark_device_dirty();
                }
            }

            encoder.endEncoding();
            return Ok(());
        }

        let entry = self
            .kv_caches
            .get(&layer_idx)
            .cloned()
            .ok_or_else(|| MetalError::InvalidOperation(format!("KV cache for layer {} not allocated", layer_idx)))?;

        if step >= entry.capacity {
            return Err(MetalError::InvalidOperation(format!(
                "Step {} exceeds KV cache capacity {} for layer {}",
                step, entry.capacity, layer_idx
            )));
        }

        let repeated_k = entry.repeated_k.clone();
        let repeated_v = entry.repeated_v.clone();

        self.prepare_tensors_for_active_cmd(&[&repeated_k, &repeated_v, &k_src, &v_src])?;

        let repeated_heads_expected = canonical_heads
            .checked_mul(group_size)
            .ok_or_else(|| MetalError::InvalidOperation("group_size overflow while expanding KV heads".into()))?;

        if repeated_k.dims()[0] != repeated_heads_expected || repeated_k.dims()[2] != head_dim {
            return Err(MetalError::DimensionMismatch {
                expected: repeated_heads_expected * head_dim,
                actual: repeated_k.dims()[0] * repeated_k.dims()[2],
            });
        }

        if v_heads != canonical_heads || v_head_dim != head_dim || v_seq_in_src != seq_in_src {
            return Err(MetalError::DimensionMismatch {
                expected: canonical_heads * head_dim,
                actual: v_heads * v_head_dim,
            });
        }

        let copy_bytes = head_dim * entry.element_size;
        let repeated_stride_elems = entry.capacity * head_dim;

        self.ensure_active_cmd_buffer()?;
        let encoder = {
            let cmd_buf = self.active_command_buffer_mut()?;
            cmd_buf
                .raw()
                .blitCommandEncoder()
                .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?
        };

        unsafe {
            for kv_head in 0..canonical_heads {
                let src_elem_index = match k_dims.len() {
                    2 => kv_head * head_dim,
                    3 => (kv_head * seq_in_src) * head_dim,
                    _ => unreachable!(),
                };
                let v_src_elem_index = match v_dims.len() {
                    2 => kv_head * head_dim,
                    3 => (kv_head * v_seq_in_src) * head_dim,
                    _ => unreachable!(),
                };

                let src_offset_k = k_src.offset + src_elem_index * entry.element_size;
                let src_offset_v = v_src.offset + v_src_elem_index * entry.element_size;

                for group in 0..group_size {
                    let repeated_head = kv_head * group_size + group;
                    let dst_elem_index = repeated_head * repeated_stride_elems + step * head_dim;

                    let dst_offset_k = repeated_k.offset + dst_elem_index * entry.element_size;
                    let dst_offset_v = repeated_v.offset + dst_elem_index * entry.element_size;

                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &k_src.buf,
                        src_offset_k,
                        &repeated_k.buf,
                        dst_offset_k,
                        copy_bytes,
                    );
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &v_src.buf,
                        src_offset_v,
                        &repeated_v.buf,
                        dst_offset_v,
                        copy_bytes,
                    );
                }
            }
        }
        encoder.endEncoding();

        self.mark_tensor_pending(&repeated_k);
        self.mark_tensor_pending(&repeated_v);

        if let Some(entry) = self.kv_caches.get_mut(&layer_idx) {
            entry.zeroing_complete = false;
        }

        Ok(())
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

    pub(crate) fn materialize_contiguous_view(&mut self, view: Tensor<T>) -> Result<Tensor<T>, MetalError> {
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
