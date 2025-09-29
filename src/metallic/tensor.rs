#![allow(dead_code)]
mod dtypes;

use super::{Context, MetalError, operation::CommandBuffer};
use crate::metallic::encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state};
use crate::metallic::kernels::elemwise_add::ElemwiseAddOp;
use crate::metallic::kernels::elemwise_div::ElemwiseDivOp;
use crate::metallic::kernels::elemwise_mul::ElemwiseMulOp;
use crate::metallic::kernels::elemwise_sub::ElemwiseSubOp;
use crate::metallic::kernels::tensors::{ArangeOp, OnesOp, RandomUniformOp};
pub use dtypes::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub type RetainedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

#[derive(Clone)]
struct ThreadSafeBuffer {
    inner: RetainedBuffer,
}

impl ThreadSafeBuffer {
    fn new(inner: RetainedBuffer) -> Self {
        Self { inner }
    }

    fn clone_inner(&self) -> RetainedBuffer {
        self.inner.clone()
    }
}

unsafe impl Send for ThreadSafeBuffer {}
unsafe impl Sync for ThreadSafeBuffer {}

impl Deref for ThreadSafeBuffer {
    type Target = RetainedBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// CPU fill threshold in MB
const DEFAULT_CPU_FILL_THRESHOLD_MB: usize = 1;

#[derive(Clone)]
struct HostAccessState {
    staging: Option<ThreadSafeBuffer>,
    staging_valid: bool,
    host_dirty: bool,
    base_offset: usize,
    region_len: usize,
}

impl HostAccessState {
    fn new(base_offset: usize, region_len: usize) -> Self {
        Self {
            staging: None,
            staging_valid: false,
            host_dirty: false,
            base_offset,
            region_len,
        }
    }

    fn region_end(&self) -> usize {
        self.base_offset.saturating_add(self.region_len)
    }
}

type HostAccessRegistry = HashMap<usize, Vec<Weak<Mutex<HostAccessState>>>>;

fn host_access_registry() -> &'static Mutex<HostAccessRegistry> {
    static REGISTRY: OnceLock<Mutex<HostAccessRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn buffer_registry_key(buffer: &RetainedBuffer) -> usize {
    Retained::as_ptr(buffer).cast::<c_void>() as usize
}

fn shared_host_access_state(buffer: &RetainedBuffer, offset: usize, len_bytes: usize) -> Arc<Mutex<HostAccessState>> {
    let mut registry = host_access_registry().lock().expect("host access registry mutex poisoned");
    let entry = registry.entry(buffer_registry_key(buffer)).or_default();

    let req_start = offset;
    let req_end = offset.saturating_add(len_bytes);
    let mut idx = 0;
    while idx < entry.len() {
        if let Some(state_arc) = entry[idx].upgrade() {
            let mut selected = false;
            {
                let mut state = state_arc.lock().expect("host access state mutex poisoned");
                let state_start = state.base_offset;
                let state_end = state.region_end();
                if req_start >= state_start && req_end <= state_end {
                    selected = true;
                } else if req_end > state_start && req_start < state_end {
                    let new_start = req_start.min(state_start);
                    let new_end = req_end.max(state_end);
                    if new_start != state_start || new_end != state_end {
                        state.base_offset = new_start;
                        state.region_len = new_end - new_start;
                        state.staging = None;
                        state.staging_valid = false;
                        state.host_dirty = false;
                    }
                    selected = true;
                }
            }

            if selected {
                return state_arc;
            }

            idx += 1;
        } else {
            entry.remove(idx);
        }
    }

    let state_arc = Arc::new(Mutex::new(HostAccessState::new(offset, len_bytes)));
    entry.push(Arc::downgrade(&state_arc));
    state_arc
}

/// A lightweight description of a tensor view that can be consumed by MPS matrix APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpsMatrixBatchView {
    pub batch: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub matrix_bytes: usize,
}

/// A small zero-copy Tensor backed by a retained Metal buffer.
/// The buffer is retained while this struct is alive. `as_slice()` provides
/// an immutable view into the underlying contents; callers must ensure
/// GPU work has completed (e.g. via command buffer wait) before reading.
/// Identifies the backing allocation strategy for a new [`Tensor`].
///
/// * [`TensorStorage::Dedicated`] allocates a fresh `MTLBuffer` from the
///   provided [`Context`]. This path supports all initialization modes and is
///   the correct choice for long-lived model weights or host-borrowed data.
/// * [`TensorStorage::Pooled`] draws memory from the context's transient bump
///   allocator. It requires a mutable context reference because it advances the
///   pool cursor and should only be used for short-lived activations.
pub enum TensorStorage<'ctx> {
    Dedicated(&'ctx Context),
    Pooled(&'ctx mut Context),
}

/// Describes how the contents of a new [`Tensor`] should be seeded.
///
/// * [`TensorInit::Uninitialized`] leaves the backing buffer untouched.
/// * [`TensorInit::CopyFrom`] copies the provided slice into the allocation.
/// * [`TensorInit::BorrowHost`] wraps an existing host slice without copying
///   and therefore requires dedicated storage.
pub enum TensorInit<'data, T: TensorElement> {
    Uninitialized,
    CopyFrom(&'data [T::Scalar]),
    BorrowHost(&'data [T::Scalar]),
}

#[derive(Clone)]
pub struct Tensor<T: TensorElement> {
    pub buf: RetainedBuffer,
    /// Shape of the tensor in elements (e.g. [batch, seq_q, dim])
    pub dims: Vec<usize>,
    /// Strides for each dimension (in elements, not bytes)
    pub strides: Vec<usize>,
    /// Data type of the tensor elements
    pub dtype: Dtype,
    /// The Metal device used to create this tensor's buffer.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Byte offset into the buffer.
    pub offset: usize,
    host_accessible: bool,
    host_access: Arc<Mutex<HostAccessState>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    /// The command buffer that must complete before this tensor's data is safe for host access.
    /// None indicates the tensor is already synchronized with the CPU.
    pub(crate) defining_cmd_buffer: Rc<RefCell<Option<CommandBuffer>>>,
    marker: PhantomData<T>,
}

pub type TensorF32 = Tensor<F32Element>;
pub type TensorF16 = Tensor<F16Element>;
pub type TensorBF16 = Tensor<BF16Element>;

impl<T: TensorElement> Tensor<T> {
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * self.dtype.size_bytes()
    }

    /// Compute a strided matrix view for tensors shaped as `[batch, rows, columns]`.
    ///
    /// The returned [`MpsMatrixBatchView`] describes how to interpret the tensor's
    /// memory layout when binding it to MPS matrix kernels. The function also
    /// supports 2-D tensors by treating them as a batch of size 1.
    pub fn as_mps_matrix_batch_view(&self) -> Result<MpsMatrixBatchView, MetalError> {
        if self.dims.len() < 2 {
            return Err(MetalError::InvalidShape(
                "MPS matrix view requires at least 2 dimensions".to_string(),
            ));
        }

        let elem_size = self.dtype.size_bytes();

        let (batch, rows, cols, row_stride_elems, matrix_stride_elems) = match self.dims.len() {
            2 => {
                let rows = self.dims[0];
                let cols = self.dims[1];
                let row_stride = if self.strides.len() == 2 { self.strides[0] } else { cols };
                let matrix_stride = rows * row_stride;
                (1, rows, cols, row_stride, matrix_stride)
            }
            3 => {
                let batch = self.dims[0];
                let rows = self.dims[1];
                let cols = self.dims[2];
                let row_stride = if self.strides.len() == 3 { self.strides[1] } else { cols };
                let matrix_stride = if self.strides.len() == 3 {
                    self.strides[0]
                } else {
                    rows * row_stride
                };
                (batch, rows, cols, row_stride, matrix_stride)
            }
            _ => {
                // Treat higher rank tensors as contiguous [batch, rows, cols] by collapsing
                // the leading dimensions into the batch dimension.
                let cols = *self
                    .dims
                    .last()
                    .ok_or_else(|| MetalError::InvalidShape("Tensor has no column dimension".to_string()))?;
                let rows = self
                    .dims
                    .iter()
                    .rev()
                    .nth(1)
                    .copied()
                    .ok_or_else(|| MetalError::InvalidShape("Tensor has no row dimension".to_string()))?;
                let batch = self.len() / (rows * cols);
                let row_stride = cols;
                let matrix_stride = rows * row_stride;
                (batch, rows, cols, row_stride, matrix_stride)
            }
        };

        let row_bytes = row_stride_elems * elem_size;
        let matrix_bytes = matrix_stride_elems * elem_size;

        if matrix_bytes < rows * row_bytes {
            return Err(MetalError::InvalidShape(
                "Tensor strides are too small for requested matrix view".to_string(),
            ));
        }

        Ok(MpsMatrixBatchView {
            batch,
            rows,
            columns: cols,
            row_bytes,
            matrix_bytes,
        })
    }

    /// Compute strides for contiguous tensor layout
    pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; dims.len()];
        if !dims.is_empty() {
            strides[dims.len() - 1] = 1;
            for i in (0..dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }
        strides
    }

    #[inline]
    fn cpu_fill_threshold_bytes() -> usize {
        DEFAULT_CPU_FILL_THRESHOLD_MB * 1024 * 1024
    }

    /// Helper function to bind a tensor to a compute encoder with correct offset
    #[inline]
    fn bind_tensor(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, tensor: &Self) {
        set_buffer(encoder, index, &tensor.buf, tensor.offset);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product::<usize>()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
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

    fn build_view(&self, dims: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
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

    fn ensure_staging_buffer(&self) -> Result<Option<RetainedBuffer>, MetalError> {
        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        let target_size = state.region_len;
        if target_size == 0 {
            state.staging = None;
            state.staging_valid = true;
            return Ok(None);
        }

        let needs_new = match state.staging.as_ref() {
            Some(buf) => buf.length() < target_size,
            None => true,
        };

        if needs_new {
            let buffer = self
                .device
                .newBufferWithLength_options(target_size, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::BufferCreationFailed(target_size))?;
            state.staging = Some(ThreadSafeBuffer::new(buffer));
            state.staging_valid = false;
        }

        Ok(state.staging.as_ref().map(ThreadSafeBuffer::clone_inner))
    }

    fn ensure_staging_for_read(&self) -> Result<(), MetalError> {
        if self.host_accessible {
            return Ok(());
        }

        let len_bytes = self.size_bytes();
        if len_bytes == 0 {
            return Ok(());
        }

        let staging = match self.ensure_staging_buffer()? {
            Some(buf) => buf,
            None => return Ok(()),
        };

        let (base_offset, region_len, skip_copy) = {
            let state = self.host_access.lock().expect("host access state mutex poisoned");
            let skip = state.host_dirty || state.staging_valid;
            (state.base_offset, state.region_len, skip)
        };

        if skip_copy || region_len == 0 {
            return Ok(());
        }

        let command_buffer = CommandBuffer::new(&self.command_queue)?;
        let encoder = command_buffer
            .raw()
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&self.buf, base_offset, &staging, 0, region_len);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.wait();

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        state.staging_valid = true;
        state.host_dirty = false;
        Ok(())
    }

    fn ensure_staging_for_write(&self) -> Result<(), MetalError> {
        if self.host_accessible {
            return Ok(());
        }

        let len_bytes = self.size_bytes();
        if len_bytes == 0 {
            return Ok(());
        }

        let staging = match self.ensure_staging_buffer()? {
            Some(buf) => buf,
            None => return Ok(()),
        };

        let (base_offset, region_len, needs_copy) = {
            let state = self.host_access.lock().expect("host access state mutex poisoned");
            let needs_copy = !state.host_dirty && !state.staging_valid;
            (state.base_offset, state.region_len, needs_copy)
        };

        if needs_copy && region_len != 0 {
            let command_buffer = CommandBuffer::new(&self.command_queue)?;
            let encoder = command_buffer
                .raw()
                .blitCommandEncoder()
                .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&self.buf, base_offset, &staging, 0, region_len);
            }
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.wait();

            let mut state = self.host_access.lock().expect("host access state mutex poisoned");
            state.staging_valid = true;
        }

        Ok(())
    }

    pub(crate) fn mark_device_dirty(&self) {
        if self.host_accessible {
            return;
        }

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        state.staging_valid = false;
        state.host_dirty = false;
    }

    pub(crate) fn flush_host_writes(&self) -> Result<(), MetalError> {
        if self.host_accessible {
            return Ok(());
        }

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        if !state.host_dirty {
            return Ok(());
        }

        if state.region_len == 0 {
            state.host_dirty = false;
            state.staging_valid = true;
            return Ok(());
        }

        let staging = state
            .staging
            .as_ref()
            .expect("staging buffer must exist when host_dirty is set")
            .clone_inner();
        let base_offset = state.base_offset;
        let region_len = state.region_len;
        drop(state);

        let command_buffer = CommandBuffer::new(&self.command_queue)?;
        let encoder = command_buffer
            .raw()
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging, 0, &self.buf, base_offset, region_len);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.wait();

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        state.host_dirty = false;
        state.staging_valid = true;
        Ok(())
    }

    fn validate_init<'data>(dims: &[usize], init: &TensorInit<'data, T>) -> Result<(), MetalError> {
        let expected_elements = dims.iter().product::<usize>();
        match init {
            TensorInit::Uninitialized => Ok(()),
            TensorInit::CopyFrom(data) | TensorInit::BorrowHost(data) => {
                if data.len() != expected_elements {
                    Err(MetalError::DimensionMismatch {
                        expected: expected_elements,
                        actual: data.len(),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    fn new_dedicated<'data>(dims: Vec<usize>, context: &Context, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
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
                let encoder = command_buffer
                    .raw()
                    .blitCommandEncoder()
                    .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging_buf, 0, &dest_buf, 0, byte_len);
                }
                encoder.endEncoding();
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

    fn new_pooled<'data>(dims: Vec<usize>, context: &mut Context, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
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
    pub fn new<'ctx, 'data>(dims: Vec<usize>, storage: TensorStorage<'ctx>, init: TensorInit<'data, T>) -> Result<Self, MetalError> {
        Self::validate_init(&dims, &init)?;

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

    fn ensure_ready(&self) {
        let pending = { self.defining_cmd_buffer.borrow().clone() };
        if let Some(cmd_buf) = pending {
            cmd_buf.commit();
            cmd_buf.wait();
            self.defining_cmd_buffer.borrow_mut().take();
        }
    }

    /// Immutable host view of the buffer. Ensure GPU work has completed before reading.
    #[inline]
    pub fn as_slice(&self) -> &[T::Scalar] {
        self.ensure_ready();
        if self.host_accessible {
            let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const T::Scalar;
            unsafe { std::slice::from_raw_parts(ptr, self.len()) }
        } else {
            self.ensure_staging_for_read()
                .expect("failed to populate staging buffer for tensor read");

            let len = self.len();
            if len == 0 {
                return &[];
            }

            let ptr = {
                let state = self.host_access.lock().expect("host access state mutex poisoned");
                let staging = state.staging.as_ref().expect("staging buffer must exist for host read");
                let byte_offset = self.offset.saturating_sub(state.base_offset);
                debug_assert!(byte_offset.is_multiple_of(std::mem::size_of::<T::Scalar>()));
                unsafe { (staging.contents().as_ptr() as *const u8).add(byte_offset) as *const T::Scalar }
            };

            unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }

    /// Copy the tensor contents to a host Vec.
    #[inline]
    pub fn to_vec(&self) -> Vec<T::Scalar> {
        self.as_slice().to_vec()
    }

    /// Synchronize given command buffers before host read. Convenience to make the read contract explicit.
    pub fn sync_before_read(buffers: &[CommandBuffer]) {
        for cb in buffers {
            cb.wait();
        }
    }

    /// Mutable host view of the buffer. Ensure no concurrent GPU access.
    pub fn as_mut_slice(&mut self) -> &mut [T::Scalar] {
        self.ensure_ready();
        if self.host_accessible {
            let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *mut T::Scalar;
            unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) }
        } else {
            self.ensure_staging_for_write()
                .expect("failed to prepare staging buffer for tensor write");

            let len = self.len();
            if len == 0 {
                return unsafe { std::slice::from_raw_parts_mut(std::ptr::NonNull::<T::Scalar>::dangling().as_ptr(), 0) };
            }

            let ptr = {
                let mut state = self.host_access.lock().expect("host access state mutex poisoned");
                let staging = state.staging.as_ref().expect("staging buffer must exist for host write").clone();
                state.host_dirty = true;
                state.staging_valid = true;
                let byte_offset = self.offset.saturating_sub(state.base_offset);
                debug_assert!(byte_offset.is_multiple_of(std::mem::size_of::<T::Scalar>()));
                drop(state);
                unsafe { (staging.contents().as_ptr() as *mut u8).add(byte_offset) as *mut T::Scalar }
            };

            unsafe { std::slice::from_raw_parts_mut(ptr, len) }
        }
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn flatten(&self) -> Self {
        self.build_view(vec![self.len()], vec![1], self.offset)
    }

    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Self, MetalError> {
        let expected_elements: usize = new_dims.iter().product();
        let actual_elements = self.len();
        if expected_elements != actual_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: actual_elements,
            });
        }
        Ok(self.build_view(new_dims.clone(), Self::compute_strides(&new_dims), self.offset))
    }

    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self, MetalError> {
        if ranges.len() > self.dims.len() {
            return Err(MetalError::InvalidShape(
                "Number of slice ranges cannot exceed tensor rank".to_string(),
            ));
        }

        // TODO: Generalize to support multi-dimensional slicing.
        if ranges.len() > 1 {
            return Err(MetalError::OperationNotSupported(
                "Slicing on more than one dimension is not yet supported.".to_string(),
            ));
        }

        let mut new_dims = self.dims.clone();
        let mut new_offset = self.offset;

        if !ranges.is_empty() {
            let range0 = &ranges[0];
            let start = range0.start;
            let end = range0.end;

            if start > end || end > self.dims[0] {
                return Err(MetalError::InvalidShape(format!(
                    "Invalid slice range {:?} for dimension 0 with size {}",
                    range0, self.dims[0]
                )));
            }

            // Update dimension for the sliced axis
            new_dims[0] = end - start;

            // Update the byte offset into the buffer
            new_offset += start * self.strides[0] * self.dtype.size_bytes();
        }

        Ok(self.build_view(new_dims.clone(), Self::compute_strides(&new_dims), new_offset))
    }

    /// Convert the tensor contents to a `Vec<f32>` using the element conversion rules.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        T::to_f32_vec(self.as_slice())
    }

    /// Create a tensor initialized from an `f32` slice by converting into the element type.
    pub fn from_f32_slice<'ctx>(dims: Vec<usize>, storage: TensorStorage<'ctx>, data: &[f32]) -> Result<Self, MetalError> {
        let converted = T::from_f32_slice(data);
        Self::new(dims, storage, TensorInit::CopyFrom(converted.as_slice()))
    }
}

impl Tensor<F32Element> {
    /// Ensure the tensor exposes a contiguous batch view suitable for batched MPS kernels.
    ///
    /// When the tensor represents a strided view into a larger cache (e.g. KV cache history)
    /// the first matrix in each batch may begin `matrix_bytes` bytes apart even if only a
    /// subset of the logical rows are active. Batched MPS operations require each matrix to
    /// be tightly packed, so this helper materializes a compact copy when padding is present.
    pub fn ensure_mps_contiguous_batch(&self, ctx: &mut Context) -> Result<(Self, MpsMatrixBatchView), MetalError> {
        let view = self.as_mps_matrix_batch_view()?;

        let needs_copy = view.batch > 1 && view.matrix_bytes != view.rows * view.row_bytes;
        if !needs_copy {
            return Ok((self.clone(), view));
        }

        let compact = Self::new(self.dims.clone(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        ctx.prepare_tensors_for_active_cmd(&[self])?;

        let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
        let encoder = command_buffer
            .raw()
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".to_string()))?;

        let copy_bytes = view.rows * view.row_bytes;
        for batch_idx in 0..view.batch {
            let src_offset = self.offset + batch_idx * view.matrix_bytes;
            let dst_offset = compact.offset + batch_idx * copy_bytes;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &self.buf,
                    src_offset,
                    &compact.buf,
                    dst_offset,
                    copy_bytes,
                );
            }
        }
        encoder.endEncoding();

        ctx.mark_tensor_pending(&compact);

        let compact_view = compact.as_mps_matrix_batch_view()?;
        Ok((compact, compact_view))
    }

    /// Allocate and zero-initialize a tensor of the given shape.
    pub fn zeros(dims: Vec<usize>, context: &mut Context, use_pool: bool) -> Result<Self, MetalError> {
        let mut tensor = if use_pool {
            Self::new(dims, TensorStorage::Pooled(context), TensorInit::Uninitialized)?
        } else {
            Self::new(dims, TensorStorage::Dedicated(&*context), TensorInit::Uninitialized)?
        };
        let size = tensor.size_bytes();

        if size <= Self::cpu_fill_threshold_bytes() {
            // CPU fill path for small tensors (use optimized fast fill)
            Self::fast_fill_f32(tensor.as_mut_slice(), 0.0f32);
        } else {
            // GPU fill path for large tensors encoded onto the active command buffer
            {
                let cmd_buf = context.active_command_buffer_mut()?;
                let encoder = cmd_buf
                    .raw()
                    .blitCommandEncoder()
                    .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;

                encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
                encoder.endEncoding();
            }
            context.mark_tensor_pending(&tensor);
        }

        Ok(tensor)
    }

    /// Create a tensor of all ones with the given shape.
    #[inline]
    pub fn ones(dims: Vec<usize>, context: &mut Context) -> Result<Self, MetalError> {
        context.call::<OnesOp>(dims)
    }

    /// Create an arange tensor (0..n as f32) with the given shape.
    pub fn arange(num_elements: usize, dims: Vec<usize>, context: &mut Context) -> Result<Self, MetalError> {
        if dims.iter().product::<usize>() != num_elements {
            return Err(MetalError::InvalidShape("dims product must match num_elements".to_string()));
        }
        let mut tensor = context.call::<ArangeOp>(num_elements)?;
        tensor.dims = dims;
        tensor.strides = Self::compute_strides(&tensor.dims);
        Ok(tensor)
    }

    /// Create a zeros tensor with the same shape.
    #[inline]
    pub fn zeros_like(&self, context: &mut Context) -> Result<Self, MetalError> {
        Self::zeros(self.dims.clone(), context, true)
    }

    /// Create a ones tensor with the same shape.
    #[inline]
    pub fn ones_like(&self, context: &mut Context) -> Result<Self, MetalError> {
        Self::ones(self.dims.clone(), context)
    }

    /// Allocate and fill a tensor with uniform random values between 0 and 1.
    #[inline]
    pub fn random_uniform(dims: Vec<usize>, context: &mut Context) -> Result<Self, MetalError> {
        // Backwards-compatible simple random uniform in [0,1)
        context.call::<RandomUniformOp>((dims, 0.0, 1.0, None))
    }

    /// Fill a new tensor with uniform random values in [min, max).
    /// Uses the device random pipeline for best performance.
    #[inline]
    pub fn random_uniform_range(dims: Vec<usize>, min: f32, max: f32, context: &mut Context) -> Result<Self, MetalError> {
        let tensor = context.call::<RandomUniformOp>((dims, min, max, None))?;
        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor using a provided command buffer (for batching).
    pub fn zeros_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Self, MetalError> {
        let tensor = Self::new(dims, TensorStorage::Pooled(context), TensorInit::Uninitialized)?;
        let size = tensor.size_bytes();

        if size <= Self::cpu_fill_threshold_bytes() {
            // CPU fill for small tensors - but since we're batching, this might not be ideal
            // For now, use GPU path for consistency
            let encoder = command_buffer.raw().blitCommandEncoder().unwrap();
            encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
            encoder.endEncoding();
        } else {
            // GPU fill for large tensors
            let encoder = command_buffer.raw().blitCommandEncoder().unwrap();
            encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
            encoder.endEncoding();
        }

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and fill a tensor with ones using a provided command buffer (for batching).
    pub fn ones_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Self, MetalError> {
        // Calculate total elements before moving dims
        let total_elements: usize = dims.iter().product();
        let tensor = Self::new(dims, TensorStorage::Pooled(context), TensorInit::Uninitialized)?;

        // Since this is batched, we still need to execute the kernel within the provided command buffer
        // So we need to get the pipeline and encode it manually
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::metallic::kernels::KernelFunction::Ones, &context.device)?;

        let encoder = command_buffer
            .raw()
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements_u32 = total_elements as u32;

        set_compute_pipeline_state(&encoder, &pipeline);
        set_buffer(&encoder, 0, &tensor.buf, tensor.offset);
        set_bytes(&encoder, 1, &total_elements_u32);

        // Dispatch threads - each thread handles 4 elements
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: total_elements.div_ceil(4),
            height: 1,
            depth: 1,
        };

        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Create an arange tensor using a provided command buffer (for batching).
    pub fn arange_batched(
        num_elements: usize,
        dims: Vec<usize>,
        command_buffer: &CommandBuffer,
        context: &mut Context,
    ) -> Result<Self, MetalError> {
        if dims.iter().product::<usize>() != num_elements {
            return Err(MetalError::InvalidShape("dims product must match num_elements".to_string()));
        }

        let tensor = Self::new(dims, TensorStorage::Pooled(context), TensorInit::Uninitialized)?;

        // Get pipeline and encode manually for batching
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::metallic::kernels::KernelFunction::Arange, &context.device)?;

        let encoder = command_buffer
            .raw()
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        set_compute_pipeline_state(&encoder, &pipeline);
        set_buffer(&encoder, 0, &tensor.buf, tensor.offset);

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: num_elements,
            height: 1,
            depth: 1,
        };

        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and fill a tensor with uniform random values using a provided command buffer (for batching).
    pub fn random_uniform_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Self, MetalError> {
        let tensor = Self::new(dims, TensorStorage::Pooled(context), TensorInit::Uninitialized)?;

        // Get pipeline and encode manually for batching with default [0, 1) range
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::metallic::kernels::KernelFunction::RandomUniform, &context.device)?;

        let encoder = command_buffer
            .raw()
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        // Use rng_seed_counter for deterministic seeding and increment
        let seed = context.rng_seed_counter as u32;
        context.rng_seed_counter += 1;

        let minv = 0.0f32; // Default min for [0, 1) range
        let scale = 1.0f32; // Default scale for [0, 1) range

        set_compute_pipeline_state(&encoder, &pipeline);
        set_buffer(&encoder, 0, &tensor.buf, tensor.offset);
        set_bytes(&encoder, 1, &seed);
        set_bytes(&encoder, 2, &minv);
        set_bytes(&encoder, 3, &scale);

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let num_elements = tensor.len();
        let grid_size = MTLSize {
            width: num_elements,
            height: 1,
            depth: 1,
        };

        dispatch_threads(&encoder, grid_size, threadgroup_size);
        encoder.endEncoding();

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor of the given shape (CPU version).
    #[deprecated(note = "Use zeros() with pooled allocation instead")]
    pub fn zeros_legacy(dims: Vec<usize>, context: &Context) -> Result<Self, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values = vec![0.0f32; num_elements];
        Self::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(&values))
    }

    /// Allocate and fill a tensor with ones (CPU version).
    #[deprecated(note = "Use ones() with pooled allocation instead")]
    pub fn ones_legacy(dims: Vec<usize>, context: &Context) -> Result<Self, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values = vec![1.0f32; num_elements];
        Self::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(&values))
    }

    /// Create an arange tensor (0..n as f32) with the given shape (CPU version).
    #[deprecated(note = "Use arange() with pooled allocation instead")]
    pub fn arange_cpu(num_elements: usize, dims: Vec<usize>, context: &Context) -> Result<Self, MetalError> {
        let v: Vec<f32> = (0..num_elements).map(|x| x as f32).collect();
        Self::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(&v))
    }

    /// Fast fill a slice of f32 with a scalar value.
    /// Uses a fast path for zeros (write_bytes) and an exponential memcpy-based
    /// fill for other values to leverage memcpy/vectorized copies instead of
    /// a per-element scalar loop.
    fn fast_fill_f32(slice: &mut [f32], value: f32) {
        if slice.is_empty() {
            return;
        }
        // Zero-specialized path: write bytes directly (very fast).
        if value == 0.0f32 {
            unsafe {
                // Write bytes: number of bytes = len * size_of::<f32>()
                std::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0u8, std::mem::size_of_val(slice));
            }
            return;
        }
        // General path: exponential copy (fill first element then double).
        unsafe {
            // Initialize the first element
            let ptr = slice.as_mut_ptr();
            std::ptr::write(ptr, value);
            let mut filled: usize = 1;
            let total = slice.len();
            while filled < total {
                let copy = std::cmp::min(filled, total - filled);
                let src = ptr;
                let dst = ptr.add(filled);
                std::ptr::copy_nonoverlapping(src, dst, copy);
                filled += copy;
            }
        }
    }

    /// Fill the tensor in-place with a scalar value (optimized).
    pub fn fill(&mut self, value: f32) {
        let slice = self.as_mut_slice();
        Self::fast_fill_f32(slice, value);
    }

    pub fn permute(&self, permute: &[usize], ctx: &mut Context) -> Result<Self, MetalError> {
        if permute.len() != self.dims.len() {
            return Err(MetalError::InvalidShape("Permutation length must match tensor rank".to_string()));
        }

        let permute_u32: Vec<u32> = permute.iter().map(|&x| x as u32).collect();

        use crate::metallic::kernels::permute::PermuteOp;
        ctx.call::<PermuteOp>((self.clone(), permute_u32))
    }

    /// Element-wise add, returns a new tensor on the same device.
    pub fn add_elem(&self, other: &Self, ctx: &mut Context) -> Result<Self, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseAddOp>((self.clone(), other.clone()))
    }

    /// Element-wise sub, returns a new tensor on the same device.
    pub fn sub_elem(&self, other: &Self, ctx: &mut Context) -> Result<Self, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseSubOp>((self.clone(), other.clone()))
    }

    /// Element-wise mul, returns a new tensor on the same device.
    pub fn mul_elem(&self, other: &Self, ctx: &mut Context) -> Result<Self, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseMulOp>((self.clone(), other.clone()))
    }

    pub fn div_elem(&self, other: &Self, ctx: &mut Context) -> Result<Self, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseDivOp>((self.clone(), other.clone()))
    }

    /// Element-wise scalar add.
    pub fn add_scalar(&self, value: f32, ctx: &mut Context) -> Result<Self, MetalError> {
        // DEBT: This is inefficient. A dedicated kernel for scalar operations would be better.
        let mut scalar_tensor = Self::zeros_like(self, ctx)?;
        scalar_tensor.fill(value);
        self.add_elem(&scalar_tensor, ctx)
    }

    fn unary_elementwise<F: Fn(f32) -> f32>(a: &Self, f: F) -> Result<Self, MetalError> {
        let byte_len = a.size_bytes();
        let buf = a
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;
        let mut out = Self {
            buf,
            dims: a.dims.clone(),
            strides: Self::compute_strides(&a.dims),
            dtype: a.dtype,
            device: a.device.clone(),
            offset: 0,
            host_accessible: true,
            host_access: Arc::new(Mutex::new(HostAccessState::new(0, byte_len))),
            command_queue: a.command_queue.clone(),
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
            marker: PhantomData,
        };
        let aslice = a.as_slice();
        let oslice = out.as_mut_slice();
        for i in 0..a.len() {
            oslice[i] = f(aslice[i]);
        }
        Ok(out)
    }

    pub fn get_batch(&self, batch_index: usize) -> Result<Self, MetalError> {
        if self.dims.len() < 3 {
            return Err(MetalError::InvalidShape("get_batch requires at least 3 dimensions".to_string()));
        }
        if batch_index >= self.dims[0] {
            return Err(MetalError::InvalidShape("batch_index out of bounds".to_string()));
        }

        let elem_size = std::mem::size_of::<f32>();
        let batch_stride_elems = if self.strides.len() == self.dims.len() && !self.strides.is_empty() {
            self.strides[0]
        } else {
            self.dims[1..].iter().product::<usize>()
        };
        let new_offset = self.offset + batch_index * batch_stride_elems * elem_size;
        let new_strides = if self.strides.len() >= 2 {
            self.strides[1..].to_vec()
        } else {
            Self::compute_strides(&self.dims[1..])
        };

        Ok(self.build_view(self.dims[1..].to_vec(), new_strides, new_offset))
    }

    /// Check tensor values for numerical stability issues
    pub fn validate_numerical_stability(&self) -> Result<(), MetalError> {
        let data = self.as_slice();
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MetalError::InvalidOperation(format!(
                    "Non-finite value detected at index {}: {} in tensor with shape {:?}",
                    i, val, self.dims
                )));
            }
            // Check for extremely large values that might cause overflow in subsequent operations
            if val.abs() > 1e6 {
                eprintln!(
                    "Warning: Very large value detected at index {}: {} in tensor with shape {:?}. This could cause numerical instability.",
                    i, val, self.dims
                );
            }
        }
        Ok(())
    }
}
