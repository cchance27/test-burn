use super::error::MetalError;
use super::pool::MemoryPool;
use super::resource_cache::ResourceCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
};

/// The main context for Metal operations.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: MemoryPool,
    // Private field for the fused softmax pipeline - not part of the public API
    pub(crate) fused_softmax_pipeline:
        Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) layernorm_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) gelu_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Context {
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
            fused_softmax_pipeline: None,
            layernorm_pipeline: None,
            gelu_pipeline: None,
        })
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
