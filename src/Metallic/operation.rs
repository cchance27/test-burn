use super::{error::MetalError, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

/// A generic GPU operation that can encode itself into a Metal command buffer.
pub trait Operation {
    /// Encode this operation into the provided command buffer.
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        cache: &mut ResourceCache,
    ) -> Result<(), MetalError>;
}

/// A light wrapper around a Metal command buffer that provides a
/// simple API to record high-level operations.
pub struct CommandBuffer {
    inner: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBuffer {
    /// Create a new command buffer from a command queue.
    pub fn new(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self, MetalError> {
        let inner = queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreationFailed)?;
        Ok(Self { inner })
    }

    /// Record an operation on this command buffer.
    pub fn record(
        &mut self,
        operation: &dyn Operation,
        cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        operation.encode(&self.inner, cache)
    }

    /// Commit the command buffer for execution.
    pub fn commit(&self) {
        self.inner.commit();
    }

    /// Wait for the command buffer to complete.
    pub fn wait(&self) {
        unsafe { self.inner.waitUntilCompleted() };
    }

    #[allow(dead_code)]
    /// Borrow the underlying command buffer if direct access is needed.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        &self.inner
    }
}
