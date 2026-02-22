use std::marker::PhantomData;

use super::{
    Foundry, storage::{Dedicated, Pooled, StorageState, View}
};
use crate::{
    error::MetalError, types::{Buffer, KernelArg, MetalResourceOptions}
};

pub mod dtypes;
pub mod enums;

pub use dtypes::{Dtype, F16, F32, Q4_0, Q4_1, Q8_0, TensorElement, U32};
pub use enums::TensorInit;

/// A strongly-typed Tensor tied to the Foundry system.
///
/// T: The element type (e.g., F32, F16).
/// S: The storage state (Dedicated, Pooled, View).
#[derive(Clone)]
pub struct Tensor<T: TensorElement, S: StorageState = Dedicated> {
    pub(crate) buffer: Buffer,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) state: S,

    // Phantom data to hold types
    _marker: PhantomData<T>,
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
        if let Err(e) = self.state.check_validity() {
            panic!("Tensor access violation: {}", e);
        }
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
        let guard = self.state.as_view_guard();
        Tensor {
            buffer: self.buffer.clone(),
            dims,
            strides,
            offset,
            state: View { guard },
            _marker: PhantomData,
        }
    }

    /// Read the tensor data back to the host.
    pub fn to_vec(&self, foundry: &Foundry) -> Vec<T::Scalar> {
        self.try_to_vec(foundry).unwrap_or_else(|e| {
            panic!("Tensor::to_vec failed: {e}");
        })
    }

    /// Read the tensor data back to the host.
    pub fn try_to_vec(&self, foundry: &Foundry) -> Result<Vec<T::Scalar>, MetalError> {
        self.state.check_validity()?;

        let num_elements = self.dims.iter().product::<usize>();
        let size_bytes = num_elements * T::DTYPE.size_bytes();

        // Dereference to ProtocolObject to match the queue context.
        let device = &foundry.device;
        let read_buffer = device
            .new_buffer(size_bytes, MetalResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(size_bytes))?;

        foundry.blit_copy_sync(&self.buffer, self.offset, &read_buffer, 0, size_bytes)?;

        // Read data
        Ok(read_buffer.read_to_vec::<T::Scalar>(num_elements))
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
            state: View { guard: None },
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
            TensorInit::Uninitialized => foundry
                .device
                .new_buffer(size_bytes, MetalResourceOptions::StorageModePrivate)
                .ok_or(MetalError::BufferCreationFailed(size_bytes))?,
            TensorInit::CopyFrom(data) => {
                // Copy with staging buffer
                let dest = foundry
                    .device
                    .new_buffer(size_bytes, MetalResourceOptions::StorageModePrivate)
                    .ok_or(MetalError::BufferCreationFailed(size_bytes))?;

                // Staging
                let src_ptr = crate::types::nonnull_void_ptr_from_slice(data, "TensorInit::CopyFrom")?;
                let staging = foundry
                    .device
                    .new_buffer_with_bytes(src_ptr, size_bytes, MetalResourceOptions::StorageModeShared)
                    .ok_or(MetalError::BufferFromBytesCreationFailed)?;

                foundry.blit_copy(&staging, 0, &dest, 0, size_bytes)?;

                dest
            }
            TensorInit::BorrowHost(data) => {
                // No copy
                let src_ptr = crate::types::nonnull_void_ptr_from_slice(data, "TensorInit::BorrowHost")?;
                let b = foundry
                    .device
                    .new_buffer_with_bytes_no_copy(src_ptr, size_bytes, MetalResourceOptions::StorageModeShared, None)
                    .ok_or(MetalError::BufferFromBytesCreationFailed)?;
                crate::types::MetalBuffer(b.0)
            }
        };

        let strides = compute_strides(&dims);
        Ok(Tensor {
            buffer,
            strides,
            dims,
            offset: 0,
            state: Dedicated,
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
            // Safety: Transmuting &[T] to &[u8]. T is TensorElement which is Pod.
            let data_u8 = crate::types::pod_as_bytes(data);
            foundry.upload_bytes(&alloc.buffer, alloc.offset, data_u8)?;
        }

        let strides = compute_strides(&dims);
        Ok(Tensor {
            buffer: alloc.buffer,
            dims,
            strides,
            offset: alloc.offset,
            state: Pooled {
                generation: alloc.generation,
                pool_generation: alloc.pool_generation,
            },
            _marker: PhantomData,
        })
    }
}

// Implement KernelArg trait for auto-binding
impl<T: TensorElement, S: StorageState> KernelArg for Tensor<T, S> {
    fn buffer(&self) -> &Buffer {
        // Enforce safety check before handing out the raw buffer
        if let Err(e) = self.state.check_validity() {
            panic!("Tensor access violation: {}", e);
        }
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
