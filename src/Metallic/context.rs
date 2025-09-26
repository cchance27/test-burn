use super::error::MetalError;
use super::operation::{CommandBuffer, Operation};
use super::pool::MemoryPool;
use super::resource_cache::ResourceCache;
use crate::metallic::kernels;
use kernels::matmul::{MatMulAlphaBetaOp, MatMulOp};
use kernels::{KernelInvocable, KernelManager};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder as _;
use objc2_metal::{MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
use rustc_hash::FxHashMap;

/// The main context for Metal operations.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
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
}

impl Context {
    /// Synchronize the command queue by submitting an empty command buffer and waiting.
    /// This ensures all previously submitted GPU work has completed.
    pub fn synchronize(&self) {
        if let Some(cb) = self.command_queue.commandBuffer() {
            cb.commit();
            unsafe { cb.waitUntilCompleted() };
        }
    }

    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
        let command_queue = device.newCommandQueue().ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device)?;
        Ok(Context {
            device,
            command_queue,
            pool,
            kernel_manager: KernelManager::new(),
            pooled_bytes_allocated: 0,
            pooled_allocations: 0,
            pool_resets: 0,
            rng_seed_counter: 0,
            kv_caches: FxHashMap::default(),
        })
    }

    pub fn with_command_buffer<F, R>(&mut self, f: F) -> Result<R, MetalError>
    where
        F: FnOnce(&mut Self, &mut CommandBuffer, &mut ResourceCache) -> Result<R, MetalError>,
    {
        let mut command_buffer = CommandBuffer::new(&self.command_queue)?;
        let mut cache = ResourceCache::new();
        let result = f(self, &mut command_buffer, &mut cache)?;
        command_buffer.commit();
        Ok(result)
    }

    pub fn call<K: KernelInvocable>(&mut self, args: K::Args) -> Result<K::Output, MetalError> {
        self.with_command_buffer(move |ctx, cmd_buf, cache| {
            let pipeline = if let Some(kernel_func) = K::function_id() {
                Some(ctx.kernel_manager.get_pipeline(kernel_func, &ctx.device)?)
            } else {
                None // For MPS operations that don't need a pipeline
            };

            // Create the operation and output tensor, passing the cache
            let (operation, output) = K::new(ctx, args, pipeline, Some(cache))?;

            // Record the operation
            cmd_buf.record(&*operation, cache)?;
            Ok(output)
        })
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
}

impl Context {
    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [seq_len, batch_heads, head_dim] (contiguous).
    pub fn alloc_kv_cache(&mut self, layer_idx: usize, seq_len: usize, batch_heads: usize, head_dim: usize) -> Result<(), MetalError> {
        // allocate tensors using the Tensor::zeros helper which uses pooled allocation internally
        let k = crate::metallic::Tensor::zeros(vec![seq_len, batch_heads, head_dim], self)?;
        let v = crate::metallic::Tensor::zeros(vec![seq_len, batch_heads, head_dim], self)?;
        // register in central cache map (store tuple: k, v, capacity)
        self.kv_caches.insert(layer_idx, (k.clone(), v.clone(), seq_len));
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
        // Lookup entry
        let entry = match self.kv_caches.get(&layer_idx) {
            Some(e) => e.clone(),
            None => {
                return Err(MetalError::InvalidOperation(format!(
                    "KV cache for layer {} not allocated",
                    layer_idx
                )));
            }
        };
        let (k_cache, v_cache, capacity_seq) = entry;
        if step >= capacity_seq {
            return Err(MetalError::InvalidOperation(format!(
                "Step {} exceeds KV cache capacity {} for layer {}",
                step, capacity_seq, layer_idx
            )));
        }

        // Validate shapes
        let bh = k_step.dims().first().cloned().unwrap_or(0);
        let hd = if k_step.dims().len() == 2 {
            k_step.dims()[1]
        } else if k_step.dims().len() == 3 {
            k_step.dims()[2]
        } else {
            0
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

        // Compute byte offsets
        let elem_size = std::mem::size_of::<f32>();
        let row_elems = expected_bh * expected_hd;
        let copy_bytes = row_elems * elem_size;
        // Destination offset in bytes = (step * row_elems) * elem_size + k_cache.offset
        let dst_base = k_cache.offset + step * row_elems * elem_size;
        let dst_base_v = v_cache.offset + step * row_elems * elem_size;

        // Source offset and size: assume k_step is tightly packed starting at its offset
        let src_offset_k = k_step.offset;
        let src_offset_v = v_step.offset;

        // Create a command buffer and blit encoder to copy slices
        let cb = self.command_queue.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
        let encoder = cb
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

        // copy K then V
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &k_step.buf,
                src_offset_k,
                &k_cache.buf,
                dst_base,
                copy_bytes,
            );
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &v_step.buf,
                src_offset_v,
                &v_cache.buf,
                dst_base_v,
                copy_bytes,
            );
        }
        encoder.endEncoding();
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }

        Ok(())
    }
}
