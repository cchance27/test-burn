#![allow(dead_code)]
use super::{Context, MetalError, operation::CommandBuffer};
use crate::metallic::encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state};
use crate::metallic::kernels::elemwise_add::ElemwiseAddOp;
use crate::metallic::kernels::elemwise_div::ElemwiseDivOp;
use crate::metallic::kernels::elemwise_mul::ElemwiseMulOp;
use crate::metallic::kernels::elemwise_sub::ElemwiseSubOp;
use crate::metallic::kernels::tensors::{ArangeOp, OnesOp, RandomUniformOp};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::ffi::c_void;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U32,
    U8,
}

impl Dtype {
    /// Size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 => 4,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
        }
    }

    /// Metal format string for this data type
    pub fn metal_format(&self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "half",
            Dtype::BF16 => "bfloat",
            Dtype::I32 => "int",
            Dtype::I64 => "long",
            Dtype::U32 => "uint",
            Dtype::U8 => "uchar",
        }
    }
}

pub type RetainedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

// CPU fill threshold in MB
const DEFAULT_CPU_FILL_THRESHOLD_MB: usize = 1;

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
pub enum TensorInit<'data> {
    Uninitialized,
    CopyFrom(&'data [f32]),
    BorrowHost(&'data [f32]),
}

#[derive(Clone)]
pub struct Tensor {
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

    /// The command buffer that must complete before this tensor's data is safe for host access.
    /// None indicates the tensor is already synchronized with the CPU.
    pub(crate) defining_cmd_buffer: Rc<RefCell<Option<CommandBuffer>>>,
}

impl Tensor {
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

    /// Ensure the tensor exposes a contiguous batch view suitable for batched MPS kernels.
    ///
    /// When the tensor represents a strided view into a larger cache (e.g. KV cache history)
    /// the first matrix in each batch may begin `matrix_bytes` bytes apart even if only a
    /// subset of the logical rows are active.  Batched MPS operations require each matrix to
    /// be tightly packed, so this helper materializes a compact copy when padding is present.
    pub fn ensure_mps_contiguous_batch(&self, ctx: &mut Context) -> Result<(Tensor, MpsMatrixBatchView), MetalError> {
        let view = self.as_mps_matrix_batch_view()?;

        let needs_copy = view.batch > 1 && view.matrix_bytes != view.rows * view.row_bytes;
        if !needs_copy {
            return Ok((self.clone(), view));
        }

        let compact = Tensor::new(
            self.dims.clone(),
            TensorStorage::Pooled(ctx),
            TensorInit::Uninitialized,
        )?;

        ctx.prepare_tensors_for_active_cmd(&[self]);

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
    fn bind_tensor(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, tensor: &Tensor) {
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
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        buf: RetainedBuffer,
        offset: usize,
    ) -> Tensor {
        Tensor {
            buf,
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype,
            device: device.clone(),
            offset,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        }
    }

    fn validate_init<'data>(dims: &[usize], init: &TensorInit<'data>) -> Result<(), MetalError> {
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

    fn new_dedicated<'data>(
        dims: Vec<usize>,
        context: &Context,
        init: TensorInit<'data>,
    ) -> Result<Tensor, MetalError> {
        match init {
            TensorInit::Uninitialized => {
                let num_elements = dims.iter().product::<usize>();
                let byte_len = num_elements * std::mem::size_of::<f32>();
                let buf = context
                    .device
                    .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                    .ok_or(MetalError::BufferCreationFailed(byte_len))?;
                Ok(Self::build_tensor(dims, Dtype::F32, &context.device, buf, 0))
            }
            TensorInit::CopyFrom(items) => {
                let byte_len = std::mem::size_of_val(items);
                let item_ptr =
                    std::ptr::NonNull::new(items.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

                let buf = unsafe {
                    context
                        .device
                        .newBufferWithBytes_length_options(item_ptr, byte_len, MTLResourceOptions::StorageModeShared)
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                };

                Ok(Self::build_tensor(dims, Dtype::F32, &context.device, buf, 0))
            }
            TensorInit::BorrowHost(data) => {
                let byte_len = std::mem::size_of_val(data);
                let item_ptr =
                    std::ptr::NonNull::new(data.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

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

                Ok(Self::build_tensor(dims, Dtype::F32, &context.device, buf, 0))
            }
        }
    }

    fn new_pooled<'data>(
        dims: Vec<usize>,
        context: &mut Context,
        init: TensorInit<'data>,
    ) -> Result<Tensor, MetalError> {
        match init {
            TensorInit::BorrowHost(_) => Err(MetalError::OperationNotSupported(
                "Borrowed host buffers are only supported with dedicated storage".to_string(),
            )),
            TensorInit::Uninitialized => Ok(context.pool.alloc_tensor(dims, Dtype::F32)?.into_tensor()),
            TensorInit::CopyFrom(data) => {
                let mut tensor = context.pool.alloc_tensor(dims, Dtype::F32)?.into_tensor();
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
    pub fn new<'ctx, 'data>(
        dims: Vec<usize>,
        storage: TensorStorage<'ctx>,
        init: TensorInit<'data>,
    ) -> Result<Tensor, MetalError> {
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
        offset: usize,
    ) -> Result<Tensor, MetalError> {
        let expected_bytes = dims.iter().product::<usize>() * std::mem::size_of::<f32>();
        if offset + expected_bytes > buffer.length() {
            return Err(MetalError::InvalidShape("buffer too small for dims/offset".into()));
        }
        Ok(Self::build_tensor(dims, dtype, device, buffer, offset))
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
    pub fn as_slice(&self) -> &[f32] {
        self.ensure_ready();
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }

    /// Copy the tensor contents to a host Vec.
    #[inline]
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    /// Synchronize given command buffers before host read. Convenience to make the read contract explicit.
    pub fn sync_before_read(buffers: &[CommandBuffer]) {
        for cb in buffers {
            cb.wait();
        }
    }

    /// Mutable host view of the buffer. Ensure no concurrent GPU access.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.ensure_ready();
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) }
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn flatten(&self) -> Tensor {
        Tensor {
            buf: self.buf.clone(),
            dims: vec![self.len()],
            strides: vec![1],
            dtype: self.dtype,
            device: self.device.clone(),
            offset: self.offset,
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
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
            dims: new_dims.clone(),
            strides: Self::compute_strides(&new_dims),
            dtype: self.dtype,
            device: self.device.clone(),
            offset: self.offset,
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
        })
    }

    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Tensor, MetalError> {
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

        Ok(Tensor {
            buf: self.buf.clone(),
            dims: new_dims.clone(),
            strides: Self::compute_strides(&new_dims), // Re-compute for correctness.
            dtype: self.dtype,
            device: self.device.clone(),
            offset: new_offset,
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
        })
    }

    /// Allocate and zero-initialize a tensor of the given shape.
    pub fn zeros(dims: Vec<usize>, context: &mut Context, use_pool: bool) -> Result<Tensor, MetalError> {
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
    pub fn ones(dims: Vec<usize>, context: &mut Context) -> Result<Tensor, MetalError> {
        context.call::<OnesOp>(dims)
    }

    /// Create an arange tensor (0..n as f32) with the given shape.
    pub fn arange(num_elements: usize, dims: Vec<usize>, context: &mut Context) -> Result<Tensor, MetalError> {
        if dims.iter().product::<usize>() != num_elements {
            return Err(MetalError::InvalidShape("dims product must match num_elements".to_string()));
        }
        let mut tensor = context.call::<ArangeOp>(num_elements)?;
        tensor.dims = dims;
        Ok(tensor)
    }

    /// Create a zeros tensor with the same shape.
    #[inline]
    pub fn zeros_like(&self, context: &mut Context) -> Result<Tensor, MetalError> {
        Self::zeros(self.dims.clone(), context, true)
    }

    /// Create a ones tensor with the same shape.
    #[inline]
    pub fn ones_like(&self, context: &mut Context) -> Result<Tensor, MetalError> {
        Self::ones(self.dims.clone(), context)
    }

    /// Allocate and fill a tensor with uniform random values between 0 and 1.
    #[inline]
    pub fn random_uniform(dims: Vec<usize>, context: &mut Context) -> Result<Tensor, MetalError> {
        // Backwards-compatible simple random uniform in [0,1)
        context.call::<RandomUniformOp>((dims, 0.0, 1.0, None))
    }

    /// Fill a new tensor with uniform random values in [min, max).
    /// Uses the device random pipeline for best performance.
    #[inline]
    pub fn random_uniform_range(dims: Vec<usize>, min: f32, max: f32, context: &mut Context) -> Result<Tensor, MetalError> {
        let tensor = context.call::<RandomUniformOp>((dims, min, max, None))?;
        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor using a provided command buffer (for batching).
    pub fn zeros_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Tensor, MetalError> {
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

        Ok(tensor)
    }

    /// Allocate and fill a tensor with ones using a provided command buffer (for batching).
    pub fn ones_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Tensor, MetalError> {
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

        Ok(tensor)
    }

    /// Create an arange tensor using a provided command buffer (for batching).
    pub fn arange_batched(
        num_elements: usize,
        dims: Vec<usize>,
        command_buffer: &CommandBuffer,
        context: &mut Context,
    ) -> Result<Tensor, MetalError> {
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

        Ok(tensor)
    }

    /// Allocate and fill a tensor with uniform random values using a provided command buffer (for batching).
    pub fn random_uniform_batched(dims: Vec<usize>, command_buffer: &CommandBuffer, context: &mut Context) -> Result<Tensor, MetalError> {
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

        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor of the given shape (CPU version).
    #[deprecated(note = "Use zeros() with pooled allocation instead")]
    pub fn zeros_legacy(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values = vec![0.0f32; num_elements];
        Self::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(&values))
    }

    /// Allocate and fill a tensor with ones (CPU version).
    #[deprecated(note = "Use ones() with pooled allocation instead")]
    pub fn ones_legacy(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values = vec![1.0f32; num_elements];
        Self::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(&values))
    }

    /// Create an arange tensor (0..n as f32) with the given shape (CPU version).
    #[deprecated(note = "Use arange() with pooled allocation instead")]
    pub fn arange_cpu(num_elements: usize, dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
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

    pub fn permute(&self, permute: &[usize], ctx: &mut Context) -> Result<Tensor, MetalError> {
        if permute.len() != self.dims.len() {
            return Err(MetalError::InvalidShape("Permutation length must match tensor rank".to_string()));
        }

        let permute_u32: Vec<u32> = permute.iter().map(|&x| x as u32).collect();

        use crate::metallic::kernels::permute::PermuteOp;
        ctx.call::<PermuteOp>((self.clone(), permute_u32))
    }

    /// Element-wise add, returns a new tensor on the same device.
    pub fn add_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseAddOp>((self.clone(), other.clone()))
    }

    /// Element-wise sub, returns a new tensor on the same device.
    pub fn sub_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseSubOp>((self.clone(), other.clone()))
    }

    /// Element-wise mul, returns a new tensor on the same device.
    pub fn mul_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseMulOp>((self.clone(), other.clone()))
    }

    pub fn div_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
        if self.dims != other.dims {
            return Err(MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseDivOp>((self.clone(), other.clone()))
    }

    /// Element-wise scalar add.
    pub fn add_scalar(&self, value: f32, ctx: &mut Context) -> Result<Tensor, MetalError> {
        // DEBT: This is inefficient. A dedicated kernel for scalar operations would be better.
        let mut scalar_tensor = Tensor::zeros_like(self, ctx)?;
        scalar_tensor.fill(value);
        self.add_elem(&scalar_tensor, ctx)
    }

    fn unary_elementwise<F: Fn(f32) -> f32>(a: &Tensor, f: F) -> Result<Tensor, MetalError> {
        let byte_len = a.size_bytes();
        let buf = a
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;
        let mut out = Tensor {
            buf,
            dims: a.dims.clone(),
            strides: Self::compute_strides(&a.dims),
            dtype: a.dtype,
            device: a.device.clone(),
            offset: 0,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        };
        let aslice = a.as_slice();
        let oslice = out.as_mut_slice();
        for i in 0..a.len() {
            oslice[i] = f(aslice[i]);
        }
        Ok(out)
    }

    pub fn get_batch(&self, batch_index: usize) -> Result<Tensor, MetalError> {
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

        Ok(Tensor {
            buf: self.buf.clone(),
            dims: self.dims[1..].to_vec(),
            strides: new_strides,
            dtype: self.dtype,
            device: self.device.clone(),
            offset: new_offset,
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
        })
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
