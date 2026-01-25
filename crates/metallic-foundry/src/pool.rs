use crate::{error::MetalError, tensor::TensorElement, types::MetalResourceOptions};

const INITIAL_CHUNK_SIZE: usize = 256 * 1024 * 1024; // 256MB
const GROWTH_FACTOR: f32 = 1.5;
const MAX_CHUNKS: usize = 16;
const DEFAULT_MAX_POOL_SIZE: usize = 5 * 1024 * 1024 * 1024; // 5GB

/// A chunk of memory in the pool.
struct PoolChunk {
    buffer: crate::types::MetalBuffer,
    cursor: usize,
    capacity: usize,
}

/// A multi-chunk, growable bump-allocator for Metal buffers.
/// This version is independent of the legacy `Context`.
pub struct MemoryPool {
    chunks: Vec<PoolChunk>,
    current_chunk: usize,
    device: crate::types::MetalDevice,
    command_queue: crate::types::MetalQueue,

    // Metrics
    pub bytes_allocated: usize,
    pub allocation_count: usize,
    pub max_pool_size: usize,
}

// Safety: Metal objects are thread-safe, and we use them via Retained pointers.
// We are managing synchronization via Foundry/Context layers mostly, but
// the Pool itself holds thread-safe Metal references.
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

/// Helper struct for returning allocation info.
/// In the Foundry system, Tensors will wrap this directly.
pub struct Allocation {
    pub buffer: crate::types::MetalBuffer,
    pub offset: usize,
    pub size: usize,
}

impl MemoryPool {
    pub fn new(device: crate::types::MetalDevice, queue: crate::types::MetalQueue) -> Result<Self, MetalError> {
        Self::with_limit(device, queue, DEFAULT_MAX_POOL_SIZE)
    }

    pub fn with_limit(device: crate::types::MetalDevice, queue: crate::types::MetalQueue, max_size: usize) -> Result<Self, MetalError> {
        if max_size == 0 {
            return Err(MetalError::OutOfMemory);
        }

        let mut pool = Self {
            chunks: Vec::new(),
            current_chunk: 0,
            device: device.clone(),
            command_queue: queue,
            bytes_allocated: 0,
            allocation_count: 0,
            max_pool_size: max_size,
        };

        let initial_chunk = INITIAL_CHUNK_SIZE.min(max_size);
        pool.allocate_new_chunk(initial_chunk)?;
        Ok(pool)
    }

    pub fn alloc<T: TensorElement>(&mut self, dims: &[usize]) -> Result<Allocation, MetalError> {
        let dtype = T::DTYPE;
        let num_elements = dims.iter().product::<usize>();
        let size_bytes = num_elements * dtype.size_bytes();
        let aligned_size = align(size_bytes, 256);

        // Try current and subsequent chunks
        for i in self.current_chunk..self.chunks.len() {
            if let Some(offset) = self.try_alloc_in_chunk(i, aligned_size) {
                self.current_chunk = i;
                self.bytes_allocated += aligned_size;
                self.allocation_count += 1;

                return Ok(Allocation {
                    buffer: self.chunks[i].buffer.clone(),
                    offset,
                    size: size_bytes,
                });
            }
        }

        // New chunk needed
        if self.chunks.len() >= MAX_CHUNKS {
            return Err(MetalError::OutOfMemory);
        }

        let current_total = self.chunks.iter().map(|c| c.capacity).sum::<usize>();
        let remaining = self.max_pool_size.checked_sub(current_total).ok_or(MetalError::OutOfMemory)?;

        if remaining < aligned_size {
            return Err(MetalError::OutOfMemory);
        }

        let last_capacity = self.chunks.last().map(|c| c.capacity).unwrap_or(0);
        let next_size = ((last_capacity as f32 * GROWTH_FACTOR) as usize).max(aligned_size).min(remaining);

        self.allocate_new_chunk(next_size)?;

        let chunk_idx = self.chunks.len() - 1;
        let offset = self.try_alloc_in_chunk(chunk_idx, aligned_size).ok_or(MetalError::OutOfMemory)?;
        self.current_chunk = chunk_idx;
        self.bytes_allocated += aligned_size;
        self.allocation_count += 1;

        Ok(Allocation {
            buffer: self.chunks[chunk_idx].buffer.clone(),
            offset,
            size: size_bytes,
        })
    }

    fn try_alloc_in_chunk(&mut self, chunk_idx: usize, size: usize) -> Option<usize> {
        let chunk = &mut self.chunks[chunk_idx];
        if chunk.cursor + size <= chunk.capacity {
            let offset = chunk.cursor;
            chunk.cursor += size;
            Some(offset)
        } else {
            None
        }
    }

    fn allocate_new_chunk(&mut self, size: usize) -> Result<(), MetalError> {
        let buffer = self
            .device
            .new_buffer(size, MetalResourceOptions::StorageModePrivate)
            .ok_or(MetalError::BufferCreationFailed(size))?;

        self.chunks.push(PoolChunk {
            buffer,
            cursor: 0,
            capacity: size,
        });
        Ok(())
    }

    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.cursor = 0;
        }
        self.current_chunk = 0;
        self.bytes_allocated = 0;
    }

    pub fn upload(&self, allocation: &Allocation, data: &[u8]) -> Result<(), MetalError> {
        let size = data.len();
        if size > allocation.size {
            return Err(MetalError::InvalidOperation("Upload data exceeds allocation size".into()));
        }

        // Host -> Shared Staging -> Private (Pooled allocation)
        let staging = self
            .device
            .new_buffer_from_slice(data, MetalResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let cmd = self.command_queue.command_buffer()?;
        let blit = cmd.blit_command_encoder()?;

        blit.copy_from_buffer(&staging, 0, &allocation.buffer, allocation.offset, size);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Ok(())
    }
}

fn align(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}
