use objc2_metal::{MTLBlitCommandEncoder as _, MTLSize};

use crate::kernels::tensors::{ArangeOp, OnesOp, RandomUniformOp};

impl<T: crate::tensor::TensorElement> super::Tensor<T> {
    /// Allocate and zero-initialize a tensor of the given shape.
    pub fn zeros(dims: Vec<usize>, context: &mut super::Context<T>, use_pool: bool) -> Result<Self, crate::MetalError> {
        let mut tensor = if use_pool {
            Self::new(
                dims.clone(),
                super::TensorStorage::Pooled(context),
                super::TensorInit::Uninitialized,
            )?
        } else {
            Self::new(
                dims.clone(),
                super::TensorStorage::Dedicated(&*context),
                super::TensorInit::Uninitialized,
            )?
        };
        let size = tensor.size_bytes();

        if size <= Self::cpu_fill_threshold_bytes() {
            // CPU fill path for small tensors (use optimized fast fill)
            Self::fast_fill_slice(tensor.as_mut_slice(), T::from_f32(0.0f32));
        } else {
            // GPU fill path for large tensors encoded onto the active command buffer
            {
                let cmd_buf = context.active_command_buffer_mut()?;
                let encoder = cmd_buf.get_blit_encoder()?;

                encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
            }
            context.mark_tensor_pending(&tensor);
        }

        Ok(tensor)
    }

    /// Create a tensor of all ones with the given shape.
    #[inline]
    pub fn ones(dims: Vec<usize>, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        context.call::<OnesOp>(dims)
    }

    /// Create an arange tensor (0..n as f32) with the given shape.
    pub fn arange(num_elements: usize, dims: Vec<usize>, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if dims.iter().product::<usize>() != num_elements {
            return Err(crate::MetalError::InvalidShape("dims product must match num_elements".to_string()));
        }
        let mut tensor = context.call::<ArangeOp>(num_elements)?;
        tensor.dims = dims;
        tensor.strides = Self::compute_strides(&tensor.dims);
        Ok(tensor)
    }

    /// Create a zeros tensor with the same shape.
    #[inline]
    pub fn zeros_like(&self, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        Self::zeros(self.dims.clone(), context, true)
    }

    /// Create a ones tensor with the same shape.
    #[inline]
    pub fn ones_like(&self, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        Self::ones(self.dims.clone(), context)
    }

    /// Allocate and fill a tensor with uniform random values between 0 and 1.
    #[inline]
    pub fn random_uniform(dims: Vec<usize>, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        // Backwards-compatible simple random uniform in [0,1)
        context.call::<RandomUniformOp>((dims, 0.0, 1.0, None))
    }

    /// Fill a new tensor with uniform random values in [min, max).
    /// Uses the device random pipeline for best performance.
    #[inline]
    pub fn random_uniform_range(dims: Vec<usize>, min: f32, max: f32, context: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        let tensor = context.call::<RandomUniformOp>((dims, min, max, None))?;
        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor using a provided command buffer (for batching).
    pub fn zeros_batched(
        dims: Vec<usize>,
        command_buffer: &super::CommandBuffer,
        context: &mut super::Context<T>,
    ) -> Result<Self, crate::MetalError> {
        let tensor = Self::new(
            dims.clone(),
            super::TensorStorage::Pooled(context),
            super::TensorInit::Uninitialized,
        )?;
        let size = tensor.size_bytes();

        if size <= Self::cpu_fill_threshold_bytes() {
            // CPU fill for small tensors - but since we're batching, this might not be ideal
            // For now, use GPU path for consistency
            let encoder = command_buffer.get_blit_encoder()?;
            encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
        } else {
            // GPU fill for large tensors
            let encoder = command_buffer.get_blit_encoder()?;
            encoder.fillBuffer_range_value(&tensor.buf, (tensor.offset..tensor.offset + size).into(), 0);
        }

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and fill a tensor with ones using a provided command buffer (for batching).
    pub fn ones_batched(
        dims: Vec<usize>,
        command_buffer: &super::CommandBuffer,
        context: &mut super::Context<T>,
    ) -> Result<Self, crate::MetalError> {
        // Calculate total elements before moving dims
        let total_elements: usize = dims.iter().product();
        let tensor = Self::new(
            dims.clone(),
            super::TensorStorage::Pooled(context),
            super::TensorInit::Uninitialized,
        )?;

        // Since this is batched, we still need to execute the kernel within the provided command buffer
        // So we need to get the pipeline and encode it manually
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::kernels::KernelFunction::Ones, T::DTYPE, &context.device)?;

        let encoder = command_buffer.get_compute_encoder()?;

        let total_elements_u32 = total_elements as u32;

        crate::encoder::set_compute_pipeline_state(&encoder, &pipeline);
        crate::encoder::set_buffer(&encoder, 0, &tensor.buf, tensor.offset);
        crate::encoder::set_bytes(&encoder, 1, &total_elements_u32);

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

        crate::encoder::dispatch_threads(&encoder, grid_size, threadgroup_size);

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Create an arange tensor using a provided command buffer (for batching).
    pub fn arange_batched(
        num_elements: usize,
        dims: Vec<usize>,
        command_buffer: &super::CommandBuffer,
        context: &mut super::Context<T>,
    ) -> Result<Self, crate::MetalError> {
        if dims.iter().product::<usize>() != num_elements {
            return Err(crate::MetalError::InvalidShape("dims product must match num_elements".to_string()));
        }

        let tensor = Self::new(
            dims.clone(),
            super::TensorStorage::Pooled(context),
            super::TensorInit::Uninitialized,
        )?;

        // Get pipeline and encode manually for batching
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::kernels::KernelFunction::Arange, T::DTYPE, &context.device)?;

        let encoder = command_buffer.get_compute_encoder()?;

        crate::encoder::set_compute_pipeline_state(&encoder, &pipeline);
        crate::encoder::set_buffer(&encoder, 0, &tensor.buf, tensor.offset);

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

        crate::encoder::dispatch_threads(&encoder, grid_size, threadgroup_size);

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and fill a tensor with uniform random values using a provided command buffer (for batching).
    pub fn random_uniform_batched(
        dims: Vec<usize>,
        command_buffer: &super::CommandBuffer,
        context: &mut super::Context<T>,
    ) -> Result<Self, crate::MetalError> {
        let tensor = Self::new(
            dims.clone(),
            super::TensorStorage::Pooled(context),
            super::TensorInit::Uninitialized,
        )?;

        // Get pipeline and encode manually for batching with default [0, 1) range
        let pipeline = context
            .kernel_manager
            .get_pipeline(crate::kernels::KernelFunction::RandomUniform, T::DTYPE, &context.device)?;

        let encoder = command_buffer.get_compute_encoder()?;

        // Use rng_seed_counter for deterministic seeding and increment
        let seed = context.rng_seed_counter as u32;
        context.rng_seed_counter += 1;

        let minv = 0.0f32; // Default min for [0, 1) range
        let scale = 1.0f32; // Default scale for [0, 1) range

        crate::encoder::set_compute_pipeline_state(&encoder, &pipeline);
        crate::encoder::set_buffer(&encoder, 0, &tensor.buf, tensor.offset);
        crate::encoder::set_bytes(&encoder, 1, &seed);
        crate::encoder::set_bytes(&encoder, 2, &minv);
        crate::encoder::set_bytes(&encoder, 3, &scale);

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

        crate::encoder::dispatch_threads(&encoder, grid_size, threadgroup_size);

        tensor.defining_cmd_buffer.borrow_mut().replace(command_buffer.clone());
        tensor.mark_device_dirty();

        Ok(tensor)
    }

    /// Allocate and zero-initialize a tensor of the given shape (CPU version).
    #[deprecated(note = "Use zeros() with pooled allocation instead")]
    pub fn zeros_legacy(dims: Vec<usize>, context: &super::Context<T>) -> Result<Self, crate::MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values: Vec<T::Scalar> = vec![T::from_f32(0.0f32); num_elements];
        Self::new(
            dims.clone(),
            super::TensorStorage::Dedicated(context),
            super::TensorInit::<T>::CopyFrom(&values),
        )
    }

    /// Allocate and fill a tensor with ones (CPU version).
    #[deprecated(note = "Use ones() with pooled allocation instead")]
    pub fn ones_legacy(dims: Vec<usize>, context: &super::Context<T>) -> Result<Self, crate::MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let values: Vec<T::Scalar> = vec![T::from_f32(1.0f32); num_elements];
        Self::new(
            dims.clone(),
            super::TensorStorage::Dedicated(context),
            super::TensorInit::<T>::CopyFrom(&values),
        )
    }

    /// Create an arange tensor (0..n as f32) with the given shape (CPU version).
    #[deprecated(note = "Use arange() with pooled allocation instead")]
    pub fn arange_cpu(num_elements: usize, dims: Vec<usize>, context: &super::Context<T>) -> Result<Self, crate::MetalError> {
        let v: Vec<T::Scalar> = (0..num_elements).map(|x| T::from_f32(x as f32)).collect();
        Self::new(
            dims.clone(),
            super::TensorStorage::Dedicated(context),
            super::TensorInit::<T>::CopyFrom(&v),
        )
    }

    /// Fast fill a slice of generic scalars with a scalar value.
    /// Uses a fast path for zeros (write_bytes) and an exponential memcpy-based
    /// fill for other values to leverage memcpy/vectorized copies instead of
    /// a per-element scalar loop.
    fn fast_fill_slice(slice: &mut [T::Scalar], value: T::Scalar) {
        if slice.is_empty() {
            return;
        }
        // Zero-specialized path: write bytes directly (very fast).
        if T::to_f32(value) == 0.0f32 {
            unsafe {
                // Write bytes: number of bytes = len * size_of::<T::Scalar>()
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
        let converted_value = T::from_f32(value);
        let slice = self.as_mut_slice();
        Self::fast_fill_slice(slice, converted_value);
    }
}
