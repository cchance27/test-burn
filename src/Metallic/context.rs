use super::error::MetalError;
use super::operation::CommandBuffer;
use super::pool::MemoryPool;
use super::resource_cache::ResourceCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

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
    pub(crate) random_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) arange_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) ones_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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
            random_pipeline: None,
            arange_pipeline: None,
            ones_pipeline: None,
        })
    }

    /// Convenience method to work with a command buffer.
    /// This allows recording multiple operations and committing them together.
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

    // Kernel to fill a buffer with uniform random numbers.
    // It uses the thread position as a seed to ensure different values for each element.
    kernel void random_uniform(
        device float *output_buffer [[buffer(0)]],
        constant uint &seed [[buffer(1)]],
        uint thread_id [[thread_position_in_grid]]
    ) {
        // Use thread_id and the provided seed to generate a unique seed for each thread
        uint2 random_seed = uint2(thread_id, seed);
        output_buffer[thread_id] = hash(random_seed);
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
