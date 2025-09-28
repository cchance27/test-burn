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

/// A small zero-copy Tensor backed by a retained Metal buffer.
/// The buffer is retained while this struct is alive. `as_slice()` provides
/// an immutable view into the underlying contents; callers must ensure
/// GPU work has completed (e.g. via command buffer wait) before reading.
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

    /// Create a tensor by copying from a host slice.
    pub fn create_tensor_from_slice(items: &[f32], dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = items.len();
        let byte_len = std::mem::size_of_val(items);
        let item_ptr = std::ptr::NonNull::new(items.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

        let buf = unsafe {
            context
                .device
                .newBufferWithBytes_length_options(item_ptr, byte_len, MTLResourceOptions::StorageModeShared)
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
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype: Dtype::F32,
            device: context.device.clone(),
            offset: 0,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        })
    }

    /// Create an uninitialized tensor of given shape. Contents are unspecified.
    pub fn create_tensor(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product();
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
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype: Dtype::F32,
            device: context.device.clone(),
            offset: 0,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        })
    }

    /// Create an uninitialized tensor of given shape using pooled memory.
    #[inline]
    pub fn create_tensor_pooled(dims: Vec<usize>, ctx: &mut Context) -> Result<Tensor, MetalError> {
        ctx.pool.alloc_tensor(dims)
    }

    /// Create a tensor view from an existing Metal buffer without copying.
    pub fn from_existing_buffer(
        buffer: RetainedBuffer,
        dims: Vec<usize>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        offset: usize,
    ) -> Result<Tensor, MetalError> {
        let expected_bytes = dims.iter().product::<usize>() * std::mem::size_of::<f32>();
        if offset + expected_bytes > buffer.length() {
            return Err(MetalError::InvalidShape("buffer too small for dims/offset".into()));
        }
        Ok(Tensor {
            buf: buffer,
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype: Dtype::F32,
            device: device.clone(),
            offset,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        })
    }

    /// Create a tensor from a host slice without copying data.
    /// The caller is responsible for ensuring the slice remains valid for the lifetime of the tensor.
    /// This is more efficient than create_tensor_from_slice for read-only data.
    pub fn from_slice_no_copy(data: &[f32], dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = data.len();
        let expected_elements = dims.iter().product::<usize>();
        if expected_elements != num_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: num_elements,
            });
        }

        let byte_len = std::mem::size_of_val(data);
        let item_ptr = std::ptr::NonNull::new(data.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

        // SAFETY: We use no-copy buffer creation, caller must ensure data lifetime
        let buf = unsafe {
            context
                .device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    item_ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                    None, // No custom deallocator needed since caller manages lifetime
                )
                .ok_or(MetalError::BufferFromBytesCreationFailed)?
        };

        Ok(Tensor {
            buf,
            dims: dims.clone(),
            strides: Self::compute_strides(&dims),
            dtype: Dtype::F32,
            device: context.device.clone(),
            offset: 0,
            defining_cmd_buffer: Rc::new(RefCell::new(None)),
        })
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
        let new_strides = self.strides.clone();
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
            dims: new_dims,
            strides: new_strides,
            dtype: self.dtype,
            device: self.device.clone(),
            offset: new_offset,
            defining_cmd_buffer: self.defining_cmd_buffer.clone(),
        })
    }

    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Tensor, MetalError> {
        if axis >= self.dims.len() {
            return Err(MetalError::InvalidShape(format!(
                "Axis {} is out of bounds for tensor with rank {}",
                axis,
                self.dims.len()
            )));
        }
        if start > self.dims[axis] || start + length > self.dims[axis] {
            return Err(MetalError::InvalidShape(format!(
                "Invalid narrow range start={} length={} for axis {} with size {}",
                start,
                length,
                axis,
                self.dims[axis]
            )));
        }

        let mut new_dims = self.dims.clone();
        new_dims[axis] = length;

        let mut tensor = self.clone();
        let stride = self.strides.get(axis).copied().unwrap_or(0);
        tensor.offset += start * stride * self.dtype.size_bytes();
        tensor.dims = new_dims;

        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor of the given shape.
    pub fn zeros(dims: Vec<usize>, context: &mut Context, use_pool: bool) -> Result<Tensor, MetalError> {
        let mut tensor = if use_pool {
            Self::create_tensor_pooled(dims, context)?
        } else {
            Self::create_tensor(dims, context)?
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
        let tensor = Self::create_tensor_pooled(dims, context)?;
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
        let tensor = Self::create_tensor_pooled(dims, context)?;

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

        let tensor = Self::create_tensor_pooled(dims, context)?;

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
        let tensor = Self::create_tensor_pooled(dims, context)?;

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
        Self::create_tensor_from_slice(&values, dims, context)
    }

    /// Allocate and fill a tensor with ones (CPU version).
    #[deprecated(note = "Use ones() with pooled allocation instead")]
    pub fn ones_legacy(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values = vec![1.0f32; num_elements];
        Self::create_tensor_from_slice(&values, dims, context)
    }

    /// Create an arange tensor (0..n as f32) with the given shape (CPU version).
    #[deprecated(note = "Use arange() with pooled allocation instead")]
    pub fn arange_cpu(num_elements: usize, dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let v: Vec<f32> = (0..num_elements).map(|x| x as f32).collect();
        Self::create_tensor_from_slice(&v, dims, context)
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

        let elem_stride = self
            .strides
            .get(0)
            .copied()
            .unwrap_or_else(|| self.dims[1..].iter().product());
        let batch_size_bytes = elem_stride * self.dtype.size_bytes();
        let new_offset = self.offset + batch_index * batch_size_bytes;

        Ok(Tensor {
            buf: self.buf.clone(),
            dims: self.dims[1..].to_vec(),
            strides: Self::compute_strides(&self.dims[1..]),
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
