use std::ptr::NonNull;

use metallic_instrumentation::{GpuProfiler, config::AppConfig};
use objc2_metal::{MTLBlitCommandEncoder as _, MTLBuffer as _, MTLDevice as _, MTLResourceOptions};

use super::{CommandBuffer, MetalError, Tensor, TensorElement};
use crate::context::GPU_PROFILER_BACKEND;

impl<T: TensorElement> Tensor<T> {
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
        command_buffer.wait();

        let mut state = self.host_access.lock().expect("host access state mutex poisoned");
        state.host_dirty = false;
        state.staging_valid = true;
        Ok(())
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
