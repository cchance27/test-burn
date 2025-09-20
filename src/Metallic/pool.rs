use super::{MetalError, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

const POOL_SIZE_BYTES: usize = 256 * 1024 * 1024; // 256MB

/// A simple bump-allocator for Metal buffers.
pub struct MemoryPool {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    cursor: usize,
}

impl MemoryPool {
    /// Creates a new memory pool with a fixed capacity.
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self, MetalError> {
        let buffer = device
            .newBufferWithLength_options(POOL_SIZE_BYTES, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(POOL_SIZE_BYTES))?;
        Ok(Self {
            buffer,
            device: device.clone(),
            cursor: 0,
        })
    }

    /// Allocates a new tensor from the pool.
    pub fn alloc_tensor(&mut self, dims: Vec<usize>) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let size_bytes = num_elements * std::mem::size_of::<f32>();
        let aligned_size = align(size_bytes, 256); // Buffers must be 256-byte aligned

        if self.cursor + aligned_size > POOL_SIZE_BYTES {
            return Err(MetalError::OutOfMemory);
        }

        let offset = self.cursor;
        self.cursor += aligned_size;

        Ok(Tensor {
            buf: self.buffer.clone(),
            dims,
            device: self.device.clone(),
            offset,
        })
    }

    /// Resets the pool cursor, invalidating all previously allocated tensors.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

fn align(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}
