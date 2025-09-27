use super::{MetalError, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::cell::RefCell;
use std::rc::Rc;

const INITIAL_CHUNK_SIZE: usize = 256 * 1024 * 1024; // 256MB
const GROWTH_FACTOR: f32 = 1.5;
const MAX_CHUNKS: usize = 16;
const MAX_POOL_SIZE: usize = 5 * 1024 * 1024 * 1024; // 5GB

/// A chunk of memory in the pool.
pub struct PoolChunk {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub cursor: usize,
    pub capacity: usize,
}

/// A multi-chunk, growable bump-allocator for Metal buffers.
pub struct MemoryPool {
    chunks: Vec<PoolChunk>,
    current_chunk: usize,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    // Metrics counters
    pub pooled_bytes_allocated: usize,
    pub pooled_allocations: usize,
    pub pool_resets: usize,
}

impl MemoryPool {
    /// Creates a new memory pool with an initial chunk.
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self, MetalError> {
        let mut pool = Self {
            chunks: Vec::new(),
            current_chunk: 0,
            device: device.clone(),
            pooled_bytes_allocated: 0,
            pooled_allocations: 0,
            pool_resets: 0,
        };
        pool.allocate_new_chunk(INITIAL_CHUNK_SIZE)?;
        Ok(pool)
    }

    /// Allocates a new tensor from the pool, growing if necessary.
    pub fn alloc_tensor(&mut self, dims: Vec<usize>) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let size_bytes = num_elements * std::mem::size_of::<f32>();
        let aligned_size = align(size_bytes, 256); // Buffers must be 256-byte aligned

        // Try to allocate in existing chunks
        for chunk_idx in self.current_chunk..self.chunks.len() {
            if let Some(offset) = self.try_alloc_in_chunk(chunk_idx, aligned_size) {
                self.current_chunk = chunk_idx;
                // Update metrics
                self.pooled_bytes_allocated += aligned_size;
                self.pooled_allocations += 1;

                return Ok(Tensor {
                    buf: self.chunks[chunk_idx].buffer.clone(),
                    dims: dims.clone(),
                    strides: Tensor::compute_strides(&dims),
                    dtype: crate::metallic::tensor::Dtype::F32,
                    device: self.device.clone(),
                    offset,
                    defining_cmd_buffer: Rc::new(RefCell::new(None)),
                });
            }
        }

        // Need to allocate a new chunk
        if self.chunks.len() >= MAX_CHUNKS {
            return Err(MetalError::OutOfMemory);
        }

        let current_total_size: usize = self.chunks.iter().map(|c| c.capacity).sum();
        let last_chunk_size = self.chunks.last().unwrap().capacity;
        let new_chunk_size = ((last_chunk_size as f32 * GROWTH_FACTOR) as usize).max(aligned_size);

        if current_total_size + new_chunk_size > MAX_POOL_SIZE {
            return Err(MetalError::OutOfMemory);
        }

        self.allocate_new_chunk(new_chunk_size)?;

        // Now allocate in the new chunk
        let chunk_idx = self.chunks.len() - 1;
        let offset = self.try_alloc_in_chunk(chunk_idx, aligned_size).unwrap();
        self.current_chunk = chunk_idx;

        // Update metrics
        self.pooled_bytes_allocated += aligned_size;
        self.pooled_allocations += 1;

        Ok(Tensor {
            buf: self.chunks[chunk_idx].buffer.clone(),
            dims: dims.clone(),
            strides: Tensor::compute_strides(&dims),
            dtype: crate::metallic::tensor::Dtype::F32,
            device: self.device.clone(),
            offset,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        })
    }

    /// Attempts to allocate in a specific chunk, returns offset if successful.
    fn try_alloc_in_chunk(&mut self, chunk_idx: usize, aligned_size: usize) -> Option<usize> {
        let chunk = &mut self.chunks[chunk_idx];
        if chunk.cursor + aligned_size <= chunk.capacity {
            let offset = chunk.cursor;
            chunk.cursor += aligned_size;
            Some(offset)
        } else {
            None
        }
    }

    /// Allocates a new chunk of the given size.
    /// Returns the number of chunks (for testing).
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the current chunk index (for testing).
    pub fn current_chunk_index(&self) -> usize {
        self.current_chunk
    }

    /// Returns the cursor positions of all chunks (for testing).
    pub fn chunk_cursors(&self) -> Vec<usize> {
        self.chunks.iter().map(|c| c.cursor).collect()
    }
    fn allocate_new_chunk(&mut self, size: usize) -> Result<(), MetalError> {
        let buffer = self
            .device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(size))?;

        self.chunks.push(PoolChunk {
            buffer,
            cursor: 0,
            capacity: size,
        });
        Ok(())
    }

    /// Resets the pool cursor, invalidating all previously allocated tensors.
    pub fn reset(&mut self) {
        self.pooled_bytes_allocated = 0;
        for chunk in &mut self.chunks {
            chunk.cursor = 0;
        }
        self.current_chunk = 0;
        self.pool_resets += 1;
    }

    pub fn total_capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity).sum()
    }

    /// Returns the total number of bytes currently in use across all chunks.
    pub fn used_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.cursor).sum()
    }
}

fn align(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}
