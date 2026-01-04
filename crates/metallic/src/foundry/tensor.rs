use std::marker::PhantomData;

// Protocol traits must be imported to use methods
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLDevice as _, MTLResourceOptions
};

use super::{
    Foundry, storage::{Dedicated, Pooled, StorageState, View}
};
use crate::{
    error::MetalError, tensor::{Dtype, TensorElement, TensorInit}, types::{Buffer, KernelArg}
};

/// A strongly-typed Tensor tied to the Foundry system.
///
/// T: The element type (e.g., F32, F16).
/// S: The storage state (Dedicated, Pooled, View).
pub struct Tensor<T: TensorElement, S: StorageState = Dedicated> {
    pub(crate) buffer: Buffer,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,

    // Phantom data to hold types
    _marker: PhantomData<(T, S)>,
}

impl<T: TensorElement, S: StorageState> Tensor<T, S> {
    /// Returns the shape of the tensor.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides of the tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the offset in bytes.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> Dtype {
        T::DTYPE
    }

    pub fn view(&self, dims: Vec<usize>, strides: Vec<usize>, offset: usize) -> Tensor<T, View> {
        Tensor {
            buffer: self.buffer.clone(),
            dims,
            strides,
            offset,
            _marker: PhantomData,
        }
    }

    /// Read the tensor data back to the host.
    pub fn to_vec(&self, foundry: &Foundry) -> Vec<T::Scalar> {
        let num_elements = self.dims.iter().product::<usize>();
        let size_bytes = num_elements * T::DTYPE.size_bytes();

        // Dereference to ProtocolObject to match the queue context.
        let device = &foundry.device;
        let read_buffer = device
            .newBufferWithLength_options(size_bytes, MTLResourceOptions::StorageModeShared)
            .expect("Failed to create read buffer");

        let cmd = foundry.queue.commandBuffer().expect("Failed to create command buffer");
        let blit = cmd.blitCommandEncoder().expect("Failed to create blit encoder");

        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&self.buffer, self.offset, &read_buffer, 0, size_bytes);
        }
        blit.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read data
        let ptr = read_buffer.contents().as_ptr() as *const T::Scalar;
        unsafe { std::slice::from_raw_parts(ptr, num_elements).to_vec() }
    }
}

impl<T: TensorElement> Tensor<T, View> {
    /// Create a Tensor view from existing parts.
    pub fn from_raw_parts(buffer: Buffer, dims: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            buffer,
            dims,
            strides,
            offset,
            _marker: PhantomData,
        }
    }
}

// Construction logic
impl<T: TensorElement> Tensor<T, Dedicated> {
    pub fn new(foundry: &mut Foundry, dims: Vec<usize>, init: TensorInit<'_, T>) -> Result<Self, MetalError> {
        Self::alloc_dedicated(foundry, dims, init)
    }
}

impl<T: TensorElement> Tensor<T, Pooled> {
    pub fn new(foundry: &mut Foundry, dims: Vec<usize>, init: TensorInit<'_, T>) -> Result<Self, MetalError> {
        Self::alloc_pooled(foundry, dims, init)
    }
}

/// Unified constructor logic
impl<T: TensorElement> Tensor<T, Dedicated> {
    fn alloc_dedicated(foundry: &mut Foundry, dims: Vec<usize>, init: TensorInit<'_, T>) -> Result<Tensor<T, Dedicated>, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let size_bytes = num_elements * T::DTYPE.size_bytes();

        let buffer = match init {
            TensorInit::Uninitialized => {
                let b = foundry
                    .device
                    .newBufferWithLength_options(size_bytes, MTLResourceOptions::StorageModePrivate)
                    .ok_or(MetalError::BufferCreationFailed(size_bytes))?;
                crate::types::MetalBuffer(b)
            }
            TensorInit::CopyFrom(data) => {
                // Copy with staging buffer
                let b = foundry
                    .device
                    .newBufferWithLength_options(size_bytes, MTLResourceOptions::StorageModePrivate)
                    .ok_or(MetalError::BufferCreationFailed(size_bytes))?;
                let dest = crate::types::MetalBuffer(b);

                // Staging
                let src_ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
                let staging = unsafe {
                    foundry
                        .device
                        .newBufferWithBytes_length_options(src_ptr, size_bytes, MTLResourceOptions::StorageModeShared)
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                };

                let cmd = foundry.queue.commandBuffer().unwrap();
                let blit = cmd.blitCommandEncoder().unwrap();
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging, 0, &dest, 0, size_bytes);
                }
                blit.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();

                dest
            }
            TensorInit::BorrowHost(data) => {
                // No copy
                let src_ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
                unsafe {
                    let b = foundry
                        .device
                        .newBufferWithBytesNoCopy_length_options_deallocator(
                            src_ptr,
                            size_bytes,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        )
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?;
                    crate::types::MetalBuffer(b)
                }
            }
        };

        let strides = compute_strides(&dims);
        Ok(Tensor {
            buffer,
            strides,
            dims,
            offset: 0,
            _marker: PhantomData,
        })
    }
}

impl<T: TensorElement> Tensor<T, Pooled> {
    fn alloc_pooled(foundry: &mut Foundry, dims: Vec<usize>, init: TensorInit<'_, T>) -> Result<Tensor<T, Pooled>, MetalError> {
        let pool = foundry
            .get_resource::<super::pool::MemoryPool>()
            .ok_or(MetalError::InvalidOperation("MemoryPool not registered in Foundry".into()))?;

        // Allocate from pool
        let alloc = pool.alloc::<T>(&dims)?;

        // Handle initialization
        if let TensorInit::CopyFrom(data) = init {
            // Safety: Transmuting &[T] to &[u8]. T is TensorElement which is Pod.
            let data_u8 = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
            // Upload data to the allocated buffer
            pool.upload(&alloc, data_u8)?;
        }

        let strides = compute_strides(&dims);
        Ok(Tensor {
            buffer: alloc.buffer,
            dims,
            strides,
            offset: alloc.offset,
            _marker: PhantomData,
        })
    }
}

// Implement KernelArg trait for auto-binding
impl<T: TensorElement, S: StorageState> KernelArg for Tensor<T, S> {
    fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn dtype(&self) -> Dtype {
        T::DTYPE
    }
    fn dims(&self) -> &[usize] {
        &self.dims
    }
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    // No-op flush for now, Foundry handles flushing via command buffer tracking eventually
    fn flush(&self) {}
}

pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut s = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = s;
        s *= dims[i];
    }
    strides
}
