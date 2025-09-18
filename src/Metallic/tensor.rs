#![allow(dead_code)]
use super::{Context, MetalError};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

pub type RetainedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// A small zero-copy Tensor backed by a retained Metal buffer.
/// The buffer is retained while this struct is alive. `as_slice()` provides
/// an immutable view into the underlying f32 contents; callers must ensure
/// GPU work has completed (e.g. via `commit_and_wait()`) before reading.
#[derive(Clone)]
pub struct Tensor {
    pub buf: RetainedBuffer,
    /// Shape of the tensor in elements (e.g. [batch, seq_q, dim])
    pub dims: Vec<usize>,
    /// The Metal device used to create this tensor's buffer.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Byte offset into the buffer.
    pub offset: usize,
}

impl Tensor {
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product::<usize>()
    }

    pub fn create_tensor_from_slice(
        items: &[f32],
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let num_elements = items.len();
        let byte_len = std::mem::size_of_val(items);
        let item_ptr =
            std::ptr::NonNull::new(items.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

        let buf = unsafe {
            context
                .device
                .newBufferWithBytes_length_options(
                    item_ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or(MetalError::BufferFromBytesCreationFailed)?
        };

        let expected_elements = dims.iter().product::<usize>();
        if expected_elements != num_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: num_elements,
            });
        }

        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    pub fn create_tensor(
        num_elements: usize,
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let byte_len = num_elements * std::mem::size_of::<f32>();
        let buf = context
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;

        let expected_elements = dims.iter().product::<usize>();
        if expected_elements != num_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: num_elements,
            });
        }
        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    pub fn as_slice(&self) -> &[f32] {
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            buf: self.buf.clone(),
            dims: vec![self.len()],
            device: self.device.clone(),
            offset: self.offset,
        }
    }

    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Tensor, MetalError> {
        let expected_elements: usize = new_dims.iter().product();
        let actual_elements = self.len();
        if expected_elements != actual_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: actual_elements,
            });
        }
        Ok(Tensor {
            buf: self.buf.clone(),
            dims: new_dims,
            device: self.device.clone(),
            offset: self.offset,
        })
    }

    pub fn zeros(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let byte_len = num_elements * std::mem::size_of::<f32>();
        let buf = context
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;

        let ptr = buf.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::write_bytes(ptr, 0, num_elements);
        }

        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    pub fn get_batch(&self, batch_index: usize) -> Result<Tensor, MetalError> {
        if self.dims.len() < 3 {
            return Err(MetalError::InvalidShape(
                "get_batch requires at least 3 dimensions".to_string(),
            ));
        }
        if batch_index >= self.dims[0] {
            return Err(MetalError::InvalidShape("batch_index out of bounds".to_string()));
        }

        let batch_size_bytes = self.dims[1..].iter().product::<usize>() * std::mem::size_of::<f32>();
        let new_offset = self.offset + batch_index * batch_size_bytes;

        Ok(Tensor {
            buf: self.buf.clone(),
            dims: self.dims[1..].to_vec(),
            device: self.device.clone(),
            offset: new_offset,
        })
    }
}
