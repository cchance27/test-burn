use super::error::MetalError;
use super::operation::CommandBuffer;
use super::pool::MemoryPool;
use super::resource_cache::ResourceCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBlitCommandEncoder as _;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder as _;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};
use rustc_hash::FxHashMap;

/// The main context for Metal operations.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
    // Metrics counters
    pub pooled_bytes_allocated: usize,
    pub pooled_allocations: usize,
    pub pool_resets: usize,
    // RNG seed counter for deterministic random generation
    pub rng_seed_counter: u64,
    // Private field for the fused softmax pipeline - not part of the public API
    pub(crate) fused_softmax_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) layernorm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gelu_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) rmsnorm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) silu_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) rope_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) kv_rearrange_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) mul_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) add_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) broadcast_add_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) permute_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) sub_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) div_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) random_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) arange_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) ones_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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
        let command_queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueCreationFailed)?;
        let pool = MemoryPool::new(&device)?;
        Ok(Context {
            device,
            command_queue,
            pool,
            pooled_bytes_allocated: 0,
            pooled_allocations: 0,
            pool_resets: 0,
            rng_seed_counter: 0,
            fused_softmax_pipeline: None,
            layernorm_pipeline: None,
            gelu_pipeline: None,
            rmsnorm_pipeline: None,
            silu_pipeline: None,
            rope_pipeline: None,
            kv_rearrange_pipeline: None,
            mul_pipeline: None,
            add_pipeline: None,
            broadcast_add_pipeline: None,
            permute_pipeline: None,
            sub_pipeline: None,
            div_pipeline: None,
            random_pipeline: None,
            arange_pipeline: None,
            ones_pipeline: None,
            kv_caches: FxHashMap::default(),
        })
    }

    pub fn with_command_buffer<F, R>(&mut self, f: F) -> Result<R, MetalError>
    where
        F: FnOnce(&mut CommandBuffer, &mut ResourceCache) -> Result<R, MetalError>,
    {
        let mut command_buffer = CommandBuffer::new(&self.command_queue)?;
        let mut cache = ResourceCache::new();
        let result = f(&mut command_buffer, &mut cache)?;
        command_buffer.commit();
        Ok(result)
    }

    pub fn matmul(
        &mut self,
        a: &super::Tensor,
        b: &super::Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<super::Tensor, MetalError> {
        let a_dims = a.dims();
        let b_dims = b.dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(MetalError::InvalidShape(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let (a_rows, a_cols) = if transpose_a {
            (a_dims[1], a_dims[0])
        } else {
            (a_dims[0], a_dims[1])
        };
        let (b_rows, b_cols) = if transpose_b {
            (b_dims[1], b_dims[0])
        } else {
            (b_dims[0], b_dims[1])
        };

        if a_cols != b_rows {
            return Err(MetalError::DimensionMismatch {
                expected: a_cols,
                actual: b_rows,
            });
        }

        let out_dims = vec![a_rows, b_cols];
        let out = super::Tensor::create_tensor_pooled(out_dims, self)?;

        let gemm_key = crate::metallic::cache_keys::MpsGemmKey {
            transpose_left: transpose_a,
            transpose_right: transpose_b,
            result_rows: a_rows,
            result_columns: b_cols,
            interior_columns: a_cols,
            alpha: 1.0,
            beta: 0.0,
        };

        let device = self.device.clone();

        self.with_command_buffer(|cmd_buf, cache| {
            let gemm = cache.get_or_create_gemm(gemm_key, &device)?;

            let left_desc = cache.get_or_create_descriptor(
                crate::metallic::cache_keys::MpsMatrixDescriptorKey {
                    rows: a.dims()[0],
                    columns: a.dims()[1],
                    row_bytes: a.dims()[1] * std::mem::size_of::<f32>(),
                },
                &device,
            )?;
            let right_desc = cache.get_or_create_descriptor(
                crate::metallic::cache_keys::MpsMatrixDescriptorKey {
                    rows: b.dims()[0],
                    columns: b.dims()[1],
                    row_bytes: b.dims()[1] * std::mem::size_of::<f32>(),
                },
                &device,
            )?;
            let result_desc = cache.get_or_create_descriptor(
                crate::metallic::cache_keys::MpsMatrixDescriptorKey {
                    rows: out.dims()[0],
                    columns: out.dims()[1],
                    row_bytes: out.dims()[1] * std::mem::size_of::<f32>(),
                },
                &device,
            )?;

            let op = crate::metallic::matmul::MatMulOperation {
                left_buf: a.buf.clone(),
                left_offset: a.offset,
                right_buf: b.buf.clone(),
                right_offset: b.offset,
                result_buf: out.buf.clone(),
                result_offset: out.offset,
                left_desc,
                right_desc,
                result_desc,
                gemm,
            };
            cmd_buf.record(&op, cache)?;
            Ok(())
        })?;

        Ok(out)
    }

    // Commit the current command buffer and wait for it to complete, then
    // create a fresh command buffer for subsequent use. This avoids using a
    // committed `MTLCommandBuffer` again which triggers the assertion seen in
    // the crash.
    //pub fn commit_and_wait(&mut self) {
    //    unsafe {
    //        self.command_buffer.commit();
    //        self.command_buffer.waitUntilCompleted();
    //    }

    //    // Create a fresh command buffer for next operations.
    //    self.command_buffer = self.command_queue.commandBuffer().unwrap();
    //}
}

impl Context {
    /// Allocate an on-device per-layer KV cache and register it in the centralized kv_caches map.
    /// Layout: [seq_len, batch_heads, head_dim] (contiguous).
    pub fn alloc_kv_cache(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        batch_heads: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        // allocate tensors using the Tensor::zeros helper which uses pooled allocation internally
        let k = crate::metallic::Tensor::zeros(vec![seq_len, batch_heads, head_dim], self)?;
        let v = crate::metallic::Tensor::zeros(vec![seq_len, batch_heads, head_dim], self)?;
        // register in central cache map (store tuple: k, v, capacity)
        self.kv_caches
            .insert(layer_idx, (k.clone(), v.clone(), seq_len));
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
        let cb = self
            .command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreationFailed)?;
        let encoder = cb
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported(
                "Blit encoder not available".into(),
            ))?;

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

/// Ensure the random compute pipeline is compiled and cached on the Context.
pub fn ensure_random_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.random_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    #include <metal_stdlib>

    using namespace metal;

    // A simple hashing-based pseudo-random number generator for Metal shaders.
    // It takes a 2D seed and produces a float in [0, 1).
    float hash(uint2 seed) {
        // Constants for hashing
        const uint k1 = 0x456789abu;
        const uint k2 = 0x89abcdefu;
        const uint k3 = 0xabcdef01u;

        // Scramble the seed
        uint n = seed.x * k1 + seed.y * k2;
        n = (n << 13) ^ n;
        n = n * (n * n * 15731u + 789221u) + 1376312589u;
        n = (n >> 13) ^ n;

        // Convert to a float in [0, 1)
        return float(n & 0x0fffffffu) / float(0x10000000u);
    }

    // Kernel to fill a buffer with uniform random numbers in [min, min+scale).
    // buffer(0) - device float *output_buffer
    // buffer(1) - constant uint &seed
    // buffer(2) - constant float &minv
    // buffer(3) - constant float &scale
    kernel void random_uniform(
        device float *output_buffer [[buffer(0)]],
        constant uint &seed [[buffer(1)]],
        constant float &minv [[buffer(2)]],
        constant float &scale [[buffer(3)]],
        uint thread_id [[thread_position_in_grid]]
    ) {
        uint2 random_seed = uint2(thread_id, seed);
        float u = hash(random_seed);
        output_buffer[thread_id] = minv + u * scale;
    }
    "#;

    let source_ns = objc2_foundation::NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = objc2_foundation::NSString::from_str("random_uniform");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.random_pipeline = Some(pipeline);
    Ok(())
}

/// Ensure the arange compute pipeline is compiled and cached on the Context.
pub fn ensure_arange_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.arange_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    #include <metal_stdlib>

    using namespace metal;

    kernel void arange_kernel(
        device float *output_buffer [[buffer(0)]],
        uint thread_id [[thread_position_in_grid]]
    ) {
        output_buffer[thread_id] = thread_id;
    }
    "#;

    let source_ns = objc2_foundation::NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = objc2_foundation::NSString::from_str("arange_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.arange_pipeline = Some(pipeline);
    Ok(())
}

/// Ensure the ones compute pipeline is compiled and cached on the Context.
pub fn ensure_ones_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.ones_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    #include <metal_stdlib>
 
    using namespace metal;
 
    // Vectorized ones kernel: each thread writes up to 4 elements.
    // buffer(0) - device float *output_buffer
    // buffer(1) - constant uint &total_elements
    kernel void ones_kernel(
        device float *output_buffer [[buffer(0)]],
        constant uint &total_elements [[buffer(1)]],
        uint thread_id [[thread_position_in_grid]]
    ) {
        uint base = thread_id * 4u;
        // Write up to 4 elements, guarding against bounds
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = base + i;
            if (idx < total_elements) {
                output_buffer[idx] = 1.0;
            }
        }
    }
    "#;

    let source_ns = objc2_foundation::NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = objc2_foundation::NSString::from_str("ones_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.ones_pipeline = Some(pipeline);
    Ok(())
}
