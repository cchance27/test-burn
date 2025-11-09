use std::{
    ptr::NonNull, rc::Rc, sync::{
        Mutex, atomic::{AtomicBool, Ordering}
    }
};

use block2::RcBlock;
use metallic_instrumentation::gpu_profiler::{CommandBufferCompletionHandler, GpuProfiler, ProfiledCommandBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize
};
use tracing::trace;

use super::{caching::ResourceCache, error::MetalError};
use crate::{
    context::GpuProfilerLabel, encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}
};

/// A generic GPU operation that can encode itself into a Metal command buffer.
pub trait Operation {
    /// Encode this operation into the provided command buffer.
    fn encode(&self, command_buffer: &CommandBuffer, cache: &mut ResourceCache) -> Result<(), MetalError>;

    /// Bind kernel arguments to the compute encoder.
    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>);
}

/// A simple test-only operation for filling buffers with zeros using the blit encoder.
/// This is intentionally kept minimal for testing batch recording and profiling infrastructure.
/// For production code, use `Tensor::zeros()` or other proper tensor generation methods.
#[cfg(test)]
pub(crate) struct TestBlitZeroFill<T: crate::TensorElement> {
    pub dst: crate::Tensor<T>,
}

#[cfg(test)]
impl<T: crate::TensorElement> Operation for TestBlitZeroFill<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_blit_encoder()?;
        let profiler_scope = GpuProfiler::profile_blit(
            command_buffer.raw(),
            &encoder,
            format!("TestBlitZeroFill@{:p}", self),
            "Metal".to_string(),
        );
        encoder.fillBuffer_range_value(&self.dst.buf, (self.dst.offset..self.dst.offset + self.dst.size_bytes()).into(), 0);
        if let Some(scope) = profiler_scope {
            scope.finish();
        }
        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        // TestBlitZeroFill uses a blit encoder, not a compute encoder, so no compute arguments to bind
    }
}

/// A light wrapper around a Metal command buffer that provides a
/// simple API to record high-level operations.
#[derive(Clone)]
pub struct CommandBuffer {
    inner: Rc<CommandBufferInner>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderType {
    MetalBlit,
    MetalCompute,
    MpsGraph,
    MpsMatrix,
}

enum ActiveEncoder {
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
}

struct CommandBufferInner {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    committed: AtomicBool,
    completed: AtomicBool,
    profiler: Mutex<Option<GpuProfiler>>,
    active_encoder: Mutex<Option<ActiveEncoder>>,
}

impl CommandBuffer {
    /// Create a new command buffer from a command queue.
    pub fn new(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self, MetalError> {
        let buffer = queue.commandBuffer().ok_or(MetalError::CommandBufferCreationFailed)?;
        Ok(Self {
            inner: Rc::new(CommandBufferInner {
                buffer,
                committed: AtomicBool::new(false),
                completed: AtomicBool::new(false),
                profiler: Mutex::new(None),
                active_encoder: Mutex::new(None),
            }),
        })
    }

    /// Record an operation on this command buffer.
    pub fn record(&self, operation: &dyn Operation, cache: &mut ResourceCache) -> Result<(), MetalError> {
        if self.inner.committed.load(Ordering::Acquire) {
            return Err(MetalError::InvalidOperation(
                "Attempted to record on a committed command buffer".to_string(),
            ));
        }
        // Encode the operation. We intentionally keep the current encoder open
        // to reduce encoder churn and improve latency. Encoders are ended when
        // switching types (blit <-> compute) or on commit().
        operation.encode(self, cache)
    }

    /// Record a batch of operations, sharing encoders across them when possible.
    ///
    /// This avoids per-operation encoder churn by deferring encoder closure until
    /// after all operations in the batch have been encoded. Profiling remains
    /// per-op, as each Operation is responsible for its own profiling scope.
    pub fn record_batch(&self, operations: &[&dyn Operation], cache: &mut ResourceCache) -> Result<(), MetalError> {
        if self.inner.committed.load(Ordering::Acquire) {
            return Err(MetalError::InvalidOperation(
                "Attempted to record on a committed command buffer".to_string(),
            ));
        }
        for &op in operations {
            op.encode(self, cache)?;
        }
        // End the current encoder once after the batch to minimize overhead and
        // ensure encoder state is flushed when callers expect a batch boundary.
        self.end_current_encoder();
        Ok(())
    }

    /// Commit the command buffer for execution.
    ///
    /// This method is idempotent: repeated calls are ignored after the first commit.
    /// This avoids "commit an already committed command buffer" errors when multiple
    /// call sites may attempt to commit the same wrapper.
    pub fn commit(&self) {
        self.end_current_encoder();
        if !self.inner.committed.swap(true, Ordering::AcqRel) {
            // If a profiler is attached, note the CPU commit instant for fallback timing
            if let Ok(guard) = self.inner.profiler.lock()
                && let Some(prof) = guard.as_ref()
            {
                prof.note_cpu_commit();
            }
            trace!(
                "CB_COMMIT profiler_present={} ",
                self.inner
                    .profiler
                    .lock()
                    .ok()
                    .and_then(|g| g.as_ref().map(|_| true))
                    .unwrap_or(false)
            );
            self.inner.buffer.commit();
        }
    }

    pub fn get_blit_encoder(&self) -> Result<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>, MetalError> {
        let mut active_encoder_guard = self.inner.active_encoder.lock().unwrap();

        if let Some(ActiveEncoder::Blit(encoder)) = active_encoder_guard.as_ref() {
            return Ok(encoder.clone());
        }

        if let Some(encoder) = active_encoder_guard.take() {
            match encoder {
                ActiveEncoder::Compute(e) => e.endEncoding(),
                ActiveEncoder::Blit(e) => e.endEncoding(),
            }
        }

        let encoder = self
            .inner
            .buffer
            .blitCommandEncoder()
            .ok_or(MetalError::OperationNotSupported("Blit encoder not available".into()))?;
        *active_encoder_guard = Some(ActiveEncoder::Blit(encoder.clone()));

        Ok(encoder)
    }

    pub fn get_compute_encoder(&self) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, MetalError> {
        let mut active_encoder_guard = self.inner.active_encoder.lock().unwrap();

        if let Some(ActiveEncoder::Compute(encoder)) = active_encoder_guard.as_ref() {
            return Ok(encoder.clone());
        }

        if let Some(encoder) = active_encoder_guard.take() {
            match encoder {
                ActiveEncoder::Compute(e) => e.endEncoding(),
                ActiveEncoder::Blit(e) => e.endEncoding(),
            }
        }

        let encoder = self
            .inner
            .buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;
        *active_encoder_guard = Some(ActiveEncoder::Compute(encoder.clone()));

        Ok(encoder)
    }

    pub fn end_current_encoder(&self) {
        if let Some(encoder) = self.inner.active_encoder.lock().unwrap().take() {
            match encoder {
                ActiveEncoder::Blit(e) => e.endEncoding(),
                ActiveEncoder::Compute(e) => e.endEncoding(),
            }
        }
    }

    /// Ensure encoder compatibility for the requested operation type.
    /// Returns true if encoder state was changed, false if compatible state already exists.
    pub fn ensure_encoder_compatibility(&self, requested_type: EncoderType) -> Result<bool, MetalError> {
        let mut active_encoder_guard = self.inner.active_encoder.lock().unwrap();

        match requested_type {
            EncoderType::MetalBlit | EncoderType::MetalCompute => {
                // Metal operations can reuse active encoders or create new ones
                // No special termination required
                Ok(false)
            }
            EncoderType::MpsGraph | EncoderType::MpsMatrix => {
                // MPS operations require no active encoder for clean command buffer state
                if active_encoder_guard.is_some() {
                    // Terminate any active Metal encoder before MPS operations
                    if let Some(encoder) = active_encoder_guard.take() {
                        match encoder {
                            ActiveEncoder::Blit(e) => e.endEncoding(),
                            ActiveEncoder::Compute(e) => e.endEncoding(),
                        }
                    }
                    Ok(true)
                } else {
                    // Command buffer is already clean, no change needed
                    Ok(false)
                }
            }
        }
    }

    /// Ensure encoder compatibility and return whether encoder was terminated.
    /// This is a convenience method that calls ensure_encoder_compatibility with logging.
    pub fn prepare_encoder_for_operation(&self, op_type: EncoderType) -> Result<bool, MetalError> {
        self.ensure_encoder_compatibility(op_type)
    }

    /// Wait for the command buffer to complete.
    pub fn wait(&self) {
        self.commit();
        if !self.inner.completed.swap(true, Ordering::AcqRel) {
            self.inner.buffer.waitUntilCompleted();
        }
        // Release any retained profiler handle once the buffer has finished executing.
        self.clear_profiler();
    }

    /// Register a callback that fires when the command buffer completes on the GPU.
    pub fn on_completed<F>(&self, callback: F)
    where
        F: Fn(&ProtocolObject<dyn MTLCommandBuffer>) + 'static,
    {
        let block = RcBlock::new(move |command_buffer: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let command_buffer = unsafe { command_buffer.as_ref() };
            callback(command_buffer);
        });

        // Hand ownership of the block over to Metal. The command buffer
        // retains the handler until completion and will release it once the
        // callback has been invoked. We intentionally leak our strong
        // reference so the block stays alive for Metal to manage.
        let raw_block = RcBlock::into_raw(block);

        unsafe {
            self.inner.buffer.addCompletedHandler(raw_block.cast());
        }
    }

    /// Borrow the underlying command buffer if direct access is needed.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        &self.inner.buffer
    }

    /// Retain the GPU profiler handle for the lifetime of this command buffer.
    pub fn retain_profiler(&self, profiler: GpuProfiler) {
        if let Ok(mut slot) = self.inner.profiler.lock() {
            *slot = Some(profiler);
        }
    }

    /// Drop the retained GPU profiler handle, if present.
    pub fn clear_profiler(&self) {
        if let Ok(mut slot) = self.inner.profiler.lock() {
            slot.take();
        }
    }

    /// Returns true if two command buffers wrap the same underlying buffer.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns whether the command buffer has completed execution.
    pub fn is_completed(&self) -> bool {
        self.inner.completed.load(Ordering::Acquire)
    }

    /// Returns whether the command buffer has been committed for execution.
    pub fn is_committed(&self) -> bool {
        self.inner.committed.load(Ordering::Acquire)
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        // Ensure any active encoder is ended to avoid Metal assertion on encoder dealloc
        // This is safe to call multiple times; `end_current_encoder` guards emptiness.
        self.end_current_encoder();
    }
}

impl ProfiledCommandBuffer for CommandBuffer {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.raw()
    }

    fn on_completed(&self, handler: CommandBufferCompletionHandler) {
        let handler = std::sync::Mutex::new(Some(handler));
        let block = RcBlock::new(move |cmd: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            if let Some(callback) = handler.lock().unwrap().take() {
                let command_buffer = unsafe { cmd.as_ref() };
                callback(command_buffer);
            }
        });
        let raw_block = RcBlock::into_raw(block);
        unsafe {
            self.inner.buffer.addCompletedHandler(raw_block.cast());
        }
    }
}

/// Helper struct to build and encode a compute kernel operation.
/// This reduces boilerplate by standardizing the common pattern of:
/// 1. Getting a compute encoder
/// 2. Setting up GPU profiling
/// 3. Setting pipeline state
/// 4. Setting buffers and bytes
/// 5. Dispatching threadgroups
///
/// # Example
/// ```ignore
/// impl Operation for MyKernel {
///     fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
///         ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
///             .pipeline(&self.pipeline)
///             .bind_args(&self.args)  // Type-safe binding of all kernel arguments
///             .dispatch_1d(self.total_elements as u32, 256);
///         Ok(())
///     }
/// }
/// ```
pub struct ComputeKernelEncoder {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _profiler_scope: Option<metallic_instrumentation::gpu_profiler::GpuProfilerScope>,
}

impl ComputeKernelEncoder {
    /// Create a new encoder and set up GPU profiling.
    #[inline]
    pub fn new(command_buffer: &CommandBuffer, profiler_label: &GpuProfilerLabel) -> Result<Self, MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;
        let profiler_scope = if crate::profiling_state::get_profiling_state() {
            let label = profiler_label.clone();
            if let Some(data) = label.data.clone() {
                GpuProfiler::profile_compute_with_data(command_buffer.raw(), &encoder, label.op_name, label.backend, data)
            } else {
                GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend)
            }
        } else {
            None
        };

        Ok(Self {
            encoder,
            _profiler_scope: profiler_scope,
        })
    }

    /// Set the compute pipeline state.
    #[inline]
    pub fn pipeline(self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) -> Self {
        set_compute_pipeline_state(&self.encoder, pipeline);
        self
    }

    /// Bind kernel arguments directly from an Operation that implements bind_to_encoder
    #[inline]
    pub fn bind_kernel<O: Operation>(self, kernel: &O) -> Self {
        kernel.bind_kernel_args(&self.encoder);
        self
    }

    /// Access the underlying encoder for custom operations
    #[inline]
    pub fn with_encoder<F>(self, f: F) -> Self
    where
        F: FnOnce(&Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    {
        f(&self.encoder);
        self
    }

    /// Set a buffer at the specified index (fallback for custom binding when needed).
    #[inline]
    pub fn buffer(self, index: usize, buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>, offset: usize) -> Self {
        set_buffer(&self.encoder, index, buffer, offset);
        self
    }

    /// Set bytes at the specified index (fallback for custom binding when needed).
    #[inline]
    pub fn bytes<T: Sized>(self, index: usize, data: &T) -> Self {
        set_bytes(&self.encoder, index, data);
        self
    }

    /// Dispatch a 1D grid of threads.
    ///
    /// # Arguments
    /// * `total_elements` - Total number of elements to process
    /// * `threads_per_threadgroup` - Number of threads per threadgroup (typically 256)
    #[inline]
    pub fn dispatch_1d(self, total_elements: u32, threads_per_threadgroup: u32) {
        let threads_per_tg = MTLSize {
            width: threads_per_threadgroup as usize,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(threads_per_threadgroup) as usize,
            height: 1,
            depth: 1,
        };
        dispatch_threadgroups(&self.encoder, groups, threads_per_tg);
    }

    /// Dispatch a 2D grid of threads.
    ///
    /// # Arguments
    /// * `width` - Width of the grid
    /// * `height` - Height of the grid
    /// * `threads_per_tg` - Threadgroup dimensions
    #[inline]
    pub fn dispatch_2d(self, width: u32, height: u32, threads_per_tg: MTLSize) {
        let groups = MTLSize {
            width: width.div_ceil(threads_per_tg.width as u32) as usize,
            height: height.div_ceil(threads_per_tg.height as u32) as usize,
            depth: 1,
        };
        dispatch_threadgroups(&self.encoder, groups, threads_per_tg);
    }

    /// Dispatch a 3D grid of threads.
    ///
    /// # Arguments
    /// * `width` - Width of the grid
    /// * `height` - Height of the grid
    /// * `depth` - Depth of the grid
    /// * `threads_per_tg` - Threadgroup dimensions
    #[inline]
    pub fn dispatch_3d(self, width: u32, height: u32, depth: u32, threads_per_tg: MTLSize) {
        let groups = MTLSize {
            width: width.div_ceil(threads_per_tg.width as u32) as usize,
            height: height.div_ceil(threads_per_tg.height as u32) as usize,
            depth: depth.div_ceil(threads_per_tg.depth as u32) as usize,
        };
        dispatch_threadgroups(&self.encoder, groups, threads_per_tg);
    }

    /// Dispatch with explicit threadgroup counts and sizes.
    /// Use this when you need full control over the dispatch parameters.
    #[inline]
    pub fn dispatch_custom(self, groups: MTLSize, threads_per_tg: MTLSize) {
        dispatch_threadgroups(&self.encoder, groups, threads_per_tg);
    }

    /// Dispatch a compute kernel using thread-level parallelism (dispatchThreads:threadsPerThreadgroup:).
    /// Use this when you need direct control over the total thread count and threadgroup size.
    #[inline]
    pub fn dispatch_threads(self, grid_size: MTLSize, threadgroup_size: MTLSize) {
        self.encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }
}
