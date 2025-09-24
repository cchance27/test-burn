use super::{Context, Tensor, error::MetalError, resource_cache::ResourceCache};
use crate::metallic::context::{
    ensure_arange_pipeline, ensure_ones_pipeline, ensure_random_pipeline,
};
use crate::metallic::encoder::{
    dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitCommandEncoder as _, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputeCommandEncoder as _,
    MTLComputePipelineState, MTLSize,
};

/// A generic GPU operation that can encode itself into a Metal command buffer.
pub trait Operation {
    /// Encode this operation into the provided command buffer.
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        cache: &mut ResourceCache,
    ) -> Result<(), MetalError>;
}

/// An operation that fills a tensor with a constant value.
pub struct FillConstant {
    pub dst: Tensor,
    pub value: f32,
    pub ones_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Operation for FillConstant {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        if self.value == 0.0 {
            // Encode blit fill for zeros
            let encoder = command_buffer.blitCommandEncoder().unwrap();
            encoder.fillBuffer_range_value(
                &self.dst.buf,
                (self.dst.offset..self.dst.offset + self.dst.size_bytes()).into(),
                0,
            );
            encoder.endEncoding();
            Ok(())
        } else if self.value == 1.0 {
            // Encode compute kernel for ones
            if let Some(pipeline) = &self.ones_pipeline {
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                set_compute_pipeline_state(&encoder, pipeline);
                set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
                // pass total elements for tail-guarding
                let num_elements = self.dst.len();
                let num_elements_u32: u32 = num_elements as u32;
                set_bytes(&encoder, 1, &num_elements_u32);
                let threads_per_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
                let threadgroup_size = MTLSize {
                    width: threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                };
                let num_vecs = num_elements.div_ceil(4);
                let grid_size = MTLSize {
                    width: num_vecs,
                    height: 1,
                    depth: 1,
                };
                dispatch_threads(&encoder, grid_size, threadgroup_size);
                encoder.endEncoding();
                Ok(())
            } else {
                Err(MetalError::OperationNotSupported(
                    "Ones pipeline not available".to_string(),
                ))
            }
        } else {
            Err(MetalError::OperationNotSupported(
                "FillConstant only supports 0.0 and 1.0".to_string(),
            ))
        }
    }
}

/// An operation that fills a tensor with sequential values (0..n).
pub struct Arange {
    pub dst: Tensor,
    pub num_elements: usize,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Operation for Arange {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
        let threads_per_threadgroup = self.pipeline.maxTotalThreadsPerThreadgroup();
        let threadgroup_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: self.num_elements,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();
        Ok(())
    }
}

/// An operation that fills a tensor with uniform random values.
pub struct RandomUniform {
    pub dst: Tensor,
    pub seed: u64,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Operation for RandomUniform {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
        set_bytes(&encoder, 1, &(self.seed as u32));
        let threads_per_threadgroup = self.pipeline.maxTotalThreadsPerThreadgroup();
        let threadgroup_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };
        let num_elements = self.dst.len();
        let grid_size = MTLSize {
            width: num_elements,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();
        Ok(())
    }
}

/// A light wrapper around a Metal command buffer that provides a
/// simple API to record high-level operations.
pub struct CommandBuffer {
    inner: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    committed: std::cell::Cell<bool>,
}

impl CommandBuffer {
    /// Create a new command buffer from a command queue.
    pub fn new(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self, MetalError> {
        let inner = queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreationFailed)?;
        Ok(Self {
            inner,
            committed: std::cell::Cell::new(false),
        })
    }

    /// Record an operation on this command buffer.
    pub fn record(
        &mut self,
        operation: &dyn Operation,
        cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // Prevent recording after commit.
        if self.committed.get() {
            return Err(MetalError::InvalidOperation(
                "Attempted to record on a committed command buffer".to_string(),
            ));
        }
        operation.encode(&self.inner, cache)
    }

    /// Commit the command buffer for execution.
    ///
    /// This method is idempotent: repeated calls are ignored after the first commit.
    /// This avoids "commit an already committed command buffer" errors when multiple
    /// call sites may attempt to commit the same wrapper.
    pub fn commit(&self) {
        if !self.committed.replace(true) {
            self.inner.commit();
        }
    }

    /// Wait for the command buffer to complete.
    pub fn wait(&self) {
        // Only wait if we actually committed the buffer.
        if self.committed.get() {
            unsafe { self.inner.waitUntilCompleted() };
        }
    }

    #[allow(dead_code)]
    /// Borrow the underlying command buffer if direct access is needed.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        &self.inner
    }
}
