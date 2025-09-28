use super::error::MetalError;
use super::instrumentation::{LatencyCollectorHandle, LatencyEvent, MemoryCollectorHandle, MemoryEvent, MemoryUsage};
use super::operation::{CommandBuffer, Operation};
use super::pool::MemoryPool;
use super::resource_cache::{CacheStats, ResourceCache};
use crate::metallic::kernels::swiglu::SwiGLUOp;
use crate::metallic::{kernels, Tensor};
use kernels::matmul::{MatMulAlphaBetaOp, MatMulOp};
use kernels::scaled_dot_product_attention::ScaledDotProductAttentionOptimizedOp;
use kernels::{KernelInvocable, KernelManager};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder as _;
use objc2_metal::{MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
use rustc_hash::FxHashMap;
use std::time::Duration;

/// The main context for Metal operations.
pub struct Context {
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
    // Keyed by layer index -> (k_cache, v_cache, capacity_seq_len)
    pub(crate) kv_caches: FxHashMap<usize, (super::Tensor, super::Tensor, usize)>,

    /// Lazily created command buffer used to batch kernel dispatches until synchronization.
    active_cmd_buffer: Option<CommandBuffer>,
    /// Resource cache associated with the active command buffer.
    active_resource_cache: Option<ResourceCache>,
    /// Optional latency collector used to report per-iteration timings.
    latency_collector: Option<LatencyCollectorHandle>,
    /// Optional memory collector used to capture detailed allocation snapshots.
    memory_collector: Option<MemoryCollectorHandle>,
}

impl Context {
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
        let pool = MemoryPool::new(&device)?;
        let kv_cache_pool = MemoryPool::new(&device)?;
        Ok(Context {
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
        })
    }

    pub fn call<K: KernelInvocable>(&mut self, args: K::Args) -> Result<Tensor, MetalError> {
        self.ensure_active_cmd_buffer()?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            Some(self.kernel_manager.get_pipeline(kernel_func, &self.device)?)
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
        let kv_cache_bytes = self.kv_caches.values().map(|(k, v, _)| k.size_bytes() + v.size_bytes()).sum();

        MemoryUsage {
            pool_used: self.pool.used_bytes(),
            pool_capacity: self.pool.total_capacity(),
            kv_used: self.kv_cache_pool.used_bytes(),
            kv_capacity: self.kv_cache_pool.total_capacity(),
            kv_cache_bytes,
        }
    }

    pub fn matmul(
        &mut self,
        a: &super::Tensor,
        b: &super::Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<super::Tensor, MetalError> {
        // Use the kernel system for matmul
        self.call::<MatMulOp>((a.clone(), b.clone(), transpose_a, transpose_b))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_alpha_beta(
        &mut self,
        a: &super::Tensor,
        b: &super::Tensor,
        result: &super::Tensor,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<super::Tensor, MetalError> {
        // Use the kernel system for matmul with alpha/beta scaling
        self.call::<MatMulAlphaBetaOp>((a.clone(), b.clone(), result.clone(), transpose_a, transpose_b, alpha, beta))
    }

    pub(crate) fn call_with_cache<K: KernelInvocable>(
        &mut self,
        args: K::Args,
        cache: &mut ResourceCache,
    ) -> Result<Tensor, MetalError> {
        self.ensure_active_cmd_buffer_internal(false)?;

        let pipeline = if let Some(kernel_func) = K::function_id() {
            Some(self.kernel_manager.get_pipeline(kernel_func, &self.device)?)
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
        a: &super::Tensor,
        b: &super::Tensor,
        result: &super::Tensor,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
        cache: &mut ResourceCache,
    ) -> Result<super::Tensor, MetalError> {
        self.call_with_cache::<MatMulAlphaBetaOp>(
            (
                a.clone(),
                b.clone(),
                result.clone(),
                transpose_a,
                transpose_b,
                alpha,
                beta,
            ),
            cache,
        )
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
}

impl Context {
    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [seq_len, batch_heads, head_dim] (contiguous).
    pub fn alloc_kv_cache(&mut self, layer_idx: usize, seq_len: usize, batch_heads: usize, head_dim: usize) -> Result<(), MetalError> {
        let dims = vec![seq_len, batch_heads, head_dim];

        // Allocate K and V tensors directly from the dedicated KV cache pool
        let k = self.kv_cache_pool.alloc_tensor(dims.clone())?;
        let v = self.kv_cache_pool.alloc_tensor(dims)?;

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

        self.kv_caches.insert(layer_idx, (k, v, seq_len));
        Ok(())
    }

    /// Write a single timestep of K and V (per-head flattened) into the per-layer cache at index `step`.
    /// - `k_step` and `v_step` must be contiguous tensors with shape [batch_heads, head_dim] or [batch_heads, 1, head_dim].
    ///   This performs a device blit copy from the source buffer into the cache at the correct offset.
    pub fn write_kv_step(
        &mut self,
        layer_idx: usize,
        step: usize,
        k_step: &crate::metallic::Tensor,
        v_step: &crate::metallic::Tensor,
    ) -> Result<(), MetalError> {
        // Lookup the canonical cache tensors and clone their handles so we can work with them
        // while issuing additional commands on `self`.
        let (k_cache_ref, v_cache_ref, capacity_seq_val) = match self.kv_caches.get(&layer_idx) {
            Some((k_cache, v_cache, capacity_seq)) => (k_cache.clone(), v_cache.clone(), *capacity_seq),
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

        // Ensure both the source tensors and the cache buffers are safe to access
        let mut k_cache = k_cache_ref;
        let mut v_cache = v_cache_ref;
        let mut k_src = k_step.clone();
        let mut v_src = v_step.clone();
        self.prepare_tensors_for_active_cmd(&mut [&mut k_cache, &mut v_cache, &mut k_src, &mut v_src]);

        // Validate shapes
        let bh = k_src.dims().first().cloned().unwrap_or(0);
        let dims = k_src.dims();
        let (seq_in_src, hd) = match dims.len() {
            2 => (1, dims[1]),
            3 => (dims[1], dims[2]),
            _ => (0, 0),
        };
        // Expected per-entry stride
        let expected_bh = k_cache.dims()[1];
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

        // Compute byte offsets
        let elem_size = std::mem::size_of::<f32>();
        let row_elems = expected_bh * expected_hd;
        let copy_bytes = row_elems * elem_size;
        // Destination offset in bytes = (step * stride_0) * elem_size + base offset
        let dst_base = k_cache.offset + step * k_cache.strides[0] * elem_size;
        let dst_base_v = v_cache.offset + step * v_cache.strides[0] * elem_size;

        // Source offset and size: assume k_step is tightly packed starting at its offset
        let src_offset_k = k_src.offset;
        let src_offset_v = v_src.offset;

        // Create a command buffer and blit encoder to copy slices
        {
            let cmd_buf = self.active_command_buffer_mut()?;
            let encoder = cmd_buf
                .raw()
                .blitCommandEncoder()
                .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

            // copy K then V
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &k_src.buf,
                    src_offset_k,
                    &k_cache.buf,
                    dst_base,
                    copy_bytes,
                );
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &v_src.buf,
                    src_offset_v,
                    &v_cache.buf,
                    dst_base_v,
                    copy_bytes,
                );
            }
            encoder.endEncoding();
        }

        self.mark_tensor_pending(&k_cache);
        self.mark_tensor_pending(&v_cache);

        Ok(())
    }

    pub fn scaled_dot_product_attention(
        &mut self,
        q: &super::Tensor,
        k: &super::Tensor,
        v: &super::Tensor,
        causal: bool,
    ) -> Result<super::Tensor, MetalError> {
        self.scaled_dot_product_attention_with_offset(q, k, v, causal, 0)
    }

    pub fn scaled_dot_product_attention_with_offset(
        &mut self,
        q: &super::Tensor,
        k: &super::Tensor,
        v: &super::Tensor,
        causal: bool,
        query_offset: usize,
    ) -> Result<super::Tensor, MetalError> {
        // Use the kernel system for SDPA
        self.call::<ScaledDotProductAttentionOptimizedOp>((
            q.clone(),
            k.clone(),
            v.clone(),
            causal,
            query_offset as u32,
        ))
    }

    /// SwiGLU implementation extracted from Qwen25 FFN block.
    /// Computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    ///
    /// # Arguments
    /// * `x_normed_flat` - Flattened input [m, d_model] where m = batch * seq
    /// * `ffn_gate` - Gate projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
    /// * `ffn_up` - Up projection weight [ff_dim, d_model] (row-major; transpose if source stored as [d_model, ff_dim])
    /// * `ffn_down` - Down projection weight [d_model, ff_dim] (row-major; transpose if source stored as [ff_dim, d_model])
    /// * `ctx` - Metal context for operations
    ///
    /// # Returns
    /// Flat output [m, d_model] (reshape externally to [batch, seq, d_model])
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    pub fn SwiGLU(
        &mut self,
        x_normed_flat: &Tensor,
        ffn_gate: &Tensor,
        ffn_gate_bias: &Tensor,
        ffn_up: &Tensor,
        ffn_up_bias: &Tensor,
        ffn_down: &Tensor,
        ffn_down_bias: &Tensor,
    ) -> Result<Tensor, MetalError> {
        // Use the kernel system to call the SwiGLU operation
        self.call::<SwiGLUOp>((
            x_normed_flat.clone(),
            ffn_gate.clone(),
            ffn_gate_bias.clone(),
            ffn_up.clone(),
            ffn_up_bias.clone(),
            ffn_down.clone(),
            ffn_down_bias.clone(),
        ))
    }
}

impl Context {
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
            self.active_resource_cache = None;
        }

        if self.active_cmd_buffer.is_none() {
            let cmd_buf = CommandBuffer::new(&self.command_queue)?;
            self.active_cmd_buffer = Some(cmd_buf);
        }

        if ensure_cache && self.active_resource_cache.is_none() {
            self.active_resource_cache = Some(ResourceCache::new());
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

    pub(crate) fn mark_tensor_pending(&self, tensor: &Tensor) {
        if let Some(active) = &self.active_cmd_buffer {
            tensor.defining_cmd_buffer.borrow_mut().replace(active.clone());
        }
    }

    fn prepare_tensor_for_active_cmd(&mut self, tensor: &mut Tensor) {
        let maybe_dep = tensor.defining_cmd_buffer.borrow().clone();
        if let Some(dep) = maybe_dep {
            if self.active_cmd_buffer.as_ref().map(|active| dep.ptr_eq(active)).unwrap_or(false) {
                return;
            }

            if dep.is_completed() {
                tensor.defining_cmd_buffer.borrow_mut().take();
                return;
            }

            dep.commit();
            dep.wait();
            tensor.defining_cmd_buffer.borrow_mut().take();
        }
    }

    pub(crate) fn prepare_tensors_for_active_cmd(&mut self, tensors: &mut [&mut Tensor]) {
        for tensor in tensors {
            self.prepare_tensor_for_active_cmd(tensor);
        }
    }
}
