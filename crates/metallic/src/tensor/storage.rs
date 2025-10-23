use std::ffi::c_void;

use objc2::rc::Retained;
use objc2_metal::{MTLBlitCommandEncoder as _, MTLBuffer as _, MTLResourceOptions};

use super::{
    Arc, CommandBuffer, Context, Dtype, HostAccessState, MTLCommandQueue, MTLDevice, MetalError, Mutex, PhantomData, ProtocolObject, Rc, RefCell, RetainedBuffer, Tensor, TensorElement, TensorInit, TensorStorage, shared_host_access_state
};

impl<T: TensorElement> Tensor<T> {
    #[inline]
    pub fn cpu_fill_threshold_bytes() -> usize {
        super::DEFAULT_CPU_FILL_THRESHOLD_MB * 1024 * 1024
    }

    fn build_tensor(
        dims: Vec<usize>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        buf: RetainedBuffer,
        offset: usize,
        host_accessible: bool,
    ) -> Self {
        let dtype = T::DTYPE;
        let len_bytes = dims.iter().product::<usize>() * dtype.size_bytes();
        let host_access = if host_accessible {
            Arc::new(Mutex::new(HostAccessState::new(offset, len_bytes)))
        } else {
            shared_host_access_state(&buf, offset, len_bytes)
        };

        Self {
            buf,
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype,
            device: device.clone(),
            offset,
            host_accessible,
            host_access,
            command_queue: command_queue.clone(),
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
            marker: PhantomData,
        }
    }

    pub fn build_view(&self, dims: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            buf: self.buf.clone(),
            dims,
            strides,
            dtype: self.dtype,
            device: self.device.clone(),
            offset,
            host_accessible: self.host_accessible,
            host_access: self.host_access.clone(),
            command_queue: self.command_queue.clone(),
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
            marker: PhantomData,
        }
    }

    fn new_dedicated<'data>(dims: Vec<usize>, context: &Context<T>, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
        match init {
            TensorInit::Uninitialized => {
                let num_elements = dims.iter().product::<usize>();
                let byte_len = num_elements * std::mem::size_of::<T::Scalar>();
                let buf = context
                    .device
                    .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
                    .ok_or(MetalError::BufferCreationFailed(byte_len))?;
                Ok(Self::build_tensor(dims, &context.device, &context.command_queue, buf, 0, false))
            }
            TensorInit::CopyFrom(items) => {
                let byte_len = std::mem::size_of_val(items);
                let item_ptr = std::ptr::NonNull::new(items.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

                let dest_buf = context
                    .device
                    .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
                    .ok_or(MetalError::BufferCreationFailed(byte_len))?;

                let staging_buf = unsafe {
                    context
                        .device
                        .newBufferWithBytes_length_options(item_ptr, byte_len, MTLResourceOptions::StorageModeShared)
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                };

                let command_buffer = CommandBuffer::new(&context.command_queue)?;
                let encoder = command_buffer.get_blit_encoder()?;

                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging_buf, 0, &dest_buf, 0, byte_len);
                }
                command_buffer.commit();
                command_buffer.wait();

                Ok(Self::build_tensor(
                    dims,
                    &context.device,
                    &context.command_queue,
                    dest_buf,
                    0,
                    false,
                ))
            }
            TensorInit::BorrowHost(data) => {
                let byte_len = std::mem::size_of_val(data);
                let item_ptr = std::ptr::NonNull::new(data.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

                let buf = unsafe {
                    context
                        .device
                        .newBufferWithBytesNoCopy_length_options_deallocator(
                            item_ptr,
                            byte_len,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        )
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                };

                Ok(Self::build_tensor(dims, &context.device, &context.command_queue, buf, 0, true))
            }
        }
    }

    fn new_pooled<'data>(dims: Vec<usize>, context: &mut Context<T>, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
        match init {
            TensorInit::BorrowHost(_) => Err(MetalError::OperationNotSupported(
                "Borrowed host buffers are only supported with dedicated storage".to_string(),
            )),
            TensorInit::Uninitialized => Ok(context.pool.alloc_tensor::<T>(dims)?.into_tensor()),
            TensorInit::CopyFrom(data) => {
                let mut tensor = context.pool.alloc_tensor::<T>(dims)?.into_tensor();
                tensor.as_mut_slice().copy_from_slice(data);
                Ok(tensor)
            }
        }
    }

    /// Create a tensor using the unified initialization path.
    ///
    /// Callers provide the desired dimensions, the allocation target, and how
    /// to seed the contents. [`TensorStorage`] selects between the dedicated
    /// device allocator and the transient memory pool, while [`TensorInit`]
    /// controls whether the buffer is left uninitialized, populated by copying
    /// from a host slice, or wrapped without copying.
    ///
    /// The returned tensor always owns Metal-backed storage; when borrowing a
    /// host slice via [`TensorInit::BorrowHost`] the lifetime of that slice must
    /// exceed the tensor's lifetime.
    pub fn new<'ctx, 'data>(dims: Vec<usize>, storage: TensorStorage<'ctx, T>, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
        init.validate(&dims)?;

        match storage {
            TensorStorage::Dedicated(context) => Self::new_dedicated(dims, context, init),
            TensorStorage::Pooled(context) => Self::new_pooled(dims, context, init),
        }
    }

    /// Create a tensor view from an existing Metal buffer without copying.
    /// The caller supplies the element type to ensure the resulting tensor
    /// reports the correct logical metadata.
    pub fn from_existing_buffer(
        buffer: RetainedBuffer,
        dims: Vec<usize>,
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        offset: usize,
        host_accessible: bool,
    ) -> Result<Self, MetalError> {
        if dtype != T::DTYPE {
            return Err(MetalError::InvalidOperation(format!(
                "dtype mismatch: expected {:?}, got {:?}",
                T::DTYPE,
                dtype
            )));
        }

        let expected_bytes = dims.iter().product::<usize>() * std::mem::size_of::<T::Scalar>();
        if offset + expected_bytes > buffer.length() {
            return Err(MetalError::InvalidShape("buffer too small for dims/offset".into()));
        }
        Ok(Self::build_tensor(dims, device, command_queue, buffer, offset, host_accessible))
    }
}
