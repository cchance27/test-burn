use std::ptr::{self, NonNull};

use metallic_instrumentation::{GpuProfiler, config::AppConfig};
use objc2_metal::{MTLBlitCommandEncoder as _, MTLBuffer as _, MTLDevice as _, MTLResourceOptions};

use super::{CommandBuffer, MetalError, Tensor, TensorElement};
use crate::context::{GPU_PROFILER_BACKEND, command_buffer_pipeline};

impl<T: TensorElement> Tensor<T> {
    fn wait_command_buffer(&self, command_buffer: &CommandBuffer) {
        let completions = command_buffer_pipeline::wait_with_pipeline(&self.command_queue, command_buffer, None);
        command_buffer_pipeline::dispatch_completions(&self.command_queue, &completions);
    }

    fn ensure_staging_buffer(&self) -> Result<Option<super::RetainedBuffer>, MetalError> {
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
            state.staging = Some(super::ThreadSafeBuffer::new(buffer));
            state.staging_valid = false;
        }

        Ok(state.staging.as_ref().map(super::ThreadSafeBuffer::clone_inner))
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
        let record_cb_timing = AppConfig::profiling_forced() || AppConfig::try_global().map(|cfg| cfg.enable_profiling).unwrap_or(true);
        if let Some(profiler) = GpuProfiler::attach(&command_buffer, record_cb_timing) {
            command_buffer.retain_profiler(profiler);
        }
        let encoder = command_buffer.get_blit_encoder()?;
        let profiler_scope = GpuProfiler::profile_blit(
            command_buffer.raw(),
            &encoder,
            super::TENSOR_STAGING_READ_OP.to_string(),
            GPU_PROFILER_BACKEND.to_string(),
        );

        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&self.buf, base_offset, &staging, 0, region_len);
        }
        if let Some(scope) = profiler_scope {
            scope.finish();
        }
        command_buffer.commit();
        self.wait_command_buffer(&command_buffer);

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
            let record_cb_timing = AppConfig::profiling_forced() || AppConfig::try_global().map(|cfg| cfg.enable_profiling).unwrap_or(true);
            if let Some(profiler) = GpuProfiler::attach(&command_buffer, record_cb_timing) {
                command_buffer.retain_profiler(profiler);
            }
            let encoder = command_buffer.get_blit_encoder()?;
            let profiler_scope = GpuProfiler::profile_blit(
                command_buffer.raw(),
                &encoder,
                super::TENSOR_STAGING_PREP_OP.to_string(),
                GPU_PROFILER_BACKEND.to_string(),
            );

            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&self.buf, base_offset, &staging, 0, region_len);
            }
            // Keep encoder open; CommandBuffer will end encoders on switch or commit
            if let Some(scope) = profiler_scope {
                scope.finish();
            }
            command_buffer.commit();
            self.wait_command_buffer(&command_buffer);

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

    /// Flush any pending host writes to the GPU buffer.
    /// This must be called before dispatching a kernel that reads from this tensor
    /// if the tensor was initialized via CopyFrom with pooled (private) storage.
    pub fn flush_host_writes(&self) -> Result<(), MetalError> {
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
        let record_cb_timing = AppConfig::profiling_forced() || AppConfig::try_global().map(|cfg| cfg.enable_profiling).unwrap_or(true);
        if let Some(profiler) = GpuProfiler::attach(&command_buffer, record_cb_timing) {
            command_buffer.retain_profiler(profiler);
        }
        let encoder = command_buffer.get_blit_encoder()?;
        let profiler_scope = GpuProfiler::profile_blit(
            command_buffer.raw(),
            &encoder,
            super::TENSOR_STAGING_FLUSH_OP.to_string(),
            GPU_PROFILER_BACKEND.to_string(),
        );

        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging, 0, &self.buf, base_offset, region_len);
        }
        if let Some(scope) = profiler_scope {
            scope.finish();
        }
        command_buffer.commit();
        self.wait_command_buffer(&command_buffer);

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        state.host_dirty = false;
        state.staging_valid = true;
        Ok(())
    }

    fn ensure_ready(&self) {
        let pending = { self.defining_cmd_buffer.borrow().clone() };
        if let Some(cmd_buf) = pending {
            self.wait_command_buffer(&cmd_buf);
            self.defining_cmd_buffer.borrow_mut().take();
        }
    }

    /// Immutable host view of the buffer. Ensure GPU work has completed before reading.
    /// NOTE: Only use when the tensor layout is contiguous; strided views must call `try_to_vec()`.
    #[inline]
    pub fn as_slice(&self) -> &[T::Scalar] {
        debug_assert!(
            self.is_contiguous(),
            "Tensor::as_slice() requires a contiguous layout. dims={:?} strides={:?}",
            self.dims(),
            self.strides
        );
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
        self.try_to_vec().expect("failed to copy tensor to host vec")
    }

    /// Fallible variant that returns a host Vec, materializing strided layouts as needed.
    /// Prefer this over `as_slice()` whenever the tensor may be a view with non-unit strides.
    pub fn try_to_vec(&self) -> Result<Vec<T::Scalar>, MetalError> {
        self.ensure_ready();

        let len = self.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        if self.is_contiguous() {
            return Ok(self.as_slice().to_vec());
        }

        let dims = self.dims.clone();
        let strides = self.strides.clone();
        let contiguous_strides = Self::compute_strides(&dims);
        let elem_size = std::mem::size_of::<T::Scalar>();

        let mut result = Vec::<T::Scalar>::with_capacity(len);
        let dst_bytes = result.as_mut_ptr() as *mut u8;

        if self.host_accessible {
            let src_ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const u8;
            copy_strided_bytes(&dims, &strides, &contiguous_strides, elem_size, src_ptr, dst_bytes);
        } else {
            self.ensure_staging_for_read()?;
            let (staging_buf, base_offset) = {
                let state = self.host_access.lock().expect("host access state mutex poisoned");
                let staging = state.staging.as_ref().expect("staging buffer must exist for host read").clone();
                let base_offset = self.offset.saturating_sub(state.base_offset);
                (staging, base_offset)
            };
            let src_ptr = unsafe { (staging_buf.contents().as_ptr() as *const u8).add(base_offset) };
            copy_strided_bytes(&dims, &strides, &contiguous_strides, elem_size, src_ptr, dst_bytes);
        }

        unsafe {
            result.set_len(len);
        }

        Ok(result)
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
                return unsafe { std::slice::from_raw_parts_mut(NonNull::<T::Scalar>::dangling().as_ptr(), 0) };
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
}

fn copy_strided_bytes(
    dims: &[usize],
    strides: &[usize],
    contiguous_strides: &[usize],
    elem_size: usize,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
) {
    if dims.is_empty() {
        unsafe {
            ptr::copy_nonoverlapping(src_ptr, dst_ptr, elem_size);
        }
        return;
    }

    copy_strided_recursive(dims, strides, contiguous_strides, elem_size, 0, src_ptr, dst_ptr, 0, 0);
}

#[allow(clippy::too_many_arguments)]
fn copy_strided_recursive(
    dims: &[usize],
    strides: &[usize],
    contiguous_strides: &[usize],
    elem_size: usize,
    level: usize,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    src_offset: usize,
    dst_offset: usize,
) {
    if level == dims.len() - 1 {
        let count = dims[level];
        let src_stride = strides[level];
        if src_stride == 1 {
            unsafe {
                ptr::copy_nonoverlapping(
                    src_ptr.add(src_offset * elem_size),
                    dst_ptr.add(dst_offset * elem_size),
                    count * elem_size,
                );
            }
        } else {
            for i in 0..count {
                unsafe {
                    ptr::copy_nonoverlapping(
                        src_ptr.add((src_offset + i * src_stride) * elem_size),
                        dst_ptr.add((dst_offset + i) * elem_size),
                        elem_size,
                    );
                }
            }
        }
        return;
    }

    let dim = dims[level];
    let src_stride = strides[level];
    let dst_stride = contiguous_strides[level];
    for i in 0..dim {
        copy_strided_recursive(
            dims,
            strides,
            contiguous_strides,
            elem_size,
            level + 1,
            src_ptr,
            dst_ptr,
            src_offset + i * src_stride,
            dst_offset + i * dst_stride,
        );
    }
}
