use super::{Tensor, error::MetalError, resource_cache::ResourceCache};
use metallic_instrumentation::gpu_profiler::{CommandBufferCompletionHandler, GpuProfiler, ProfiledCommandBuffer};
use tracing::trace;

use crate::{
    TensorElement,
    encoder::{dispatch_threads, set_buffer, set_bytes, set_compute_pipeline_state},
};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};
use std::{
    ptr::NonNull,
    rc::Rc,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

/// A generic GPU operation that can encode itself into a Metal command buffer.
pub trait Operation {
    /// Encode this operation into the provided command buffer.
    fn encode(&self, command_buffer: &CommandBuffer, cache: &mut ResourceCache) -> Result<(), MetalError>;
}

//TODO: Aren't these operations supposed to be in kernels?

/// An operation that fills a tensor with a constant value.
pub struct FillConstant<T: TensorElement> {
    pub dst: Tensor<T>,
    pub value: f32,
    pub ones_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl<T: TensorElement> Operation for FillConstant<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        if self.value == 0.0 {
            // Encode blit fill for zeros
            let encoder = command_buffer.get_blit_encoder()?;
            let profiler_scope = GpuProfiler::profile_blit(
                command_buffer.raw(),
                &encoder,
                format!("FillConstantZero@{:p}", self),
                "Metal".to_string(),
            );
            encoder.fillBuffer_range_value(&self.dst.buf, (self.dst.offset..self.dst.offset + self.dst.size_bytes()).into(), 0);
            if let Some(scope) = profiler_scope {
                scope.finish();
            }
            Ok(())
        } else if self.value == 1.0 {
            // Encode compute kernel for ones
            if let Some(pipeline) = &self.ones_pipeline {
                let encoder = command_buffer.get_compute_encoder()?;
                let profiler_scope = GpuProfiler::profile_compute(
                    command_buffer.raw(),
                    &encoder,
                    format!("FillConstantOnes@{:p}", self),
                    "Metal".to_string(),
                );
                set_compute_pipeline_state(&encoder, pipeline);
                set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
                // pass total elements for tail-guarding
                let num_elements = self.dst.len();
                let num_elements_u32: u32 = num_elements as u32;
                set_bytes(&encoder, 1, &num_elements_u32);
                let threads_per_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
                let threadgroup_size = MTLSize {
                    width: threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                };
                let num_vecs = num_elements.div_ceil(4);
                let grid_size = MTLSize {
                    width: num_vecs,
                    height: 1,
                    depth: 1,
                };
                dispatch_threads(&encoder, grid_size, threadgroup_size);
                if let Some(scope) = profiler_scope {
                    scope.finish();
                }
                Ok(())
            } else {
                Err(MetalError::OperationNotSupported("Ones pipeline not available".to_string()))
            }
        } else {
            Err(MetalError::OperationNotSupported(
                "FillConstant only supports 0.0 and 1.0".to_string(),
            ))
        }
    }
}

/// An operation that fills a tensor with sequential values (0..n).
pub struct Arange<T: TensorElement> {
    pub dst: Tensor<T>,
    pub num_elements: usize,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl<T: TensorElement> Operation for Arange<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
        let threads_per_threadgroup = self.pipeline.maxTotalThreadsPerThreadgroup();
        let threadgroup_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: self.num_elements,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid_size, threadgroup_size);
        Ok(())
    }
}

/// An operation that fills a tensor with uniform random values.
pub struct RandomUniform<T: TensorElement> {
    pub dst: Tensor<T>,
    pub seed: u64,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl<T: TensorElement> Operation for RandomUniform<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;
        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.dst.buf, self.dst.offset);
        set_bytes(&encoder, 1, &(self.seed as u32));
        let threads_per_threadgroup = self.pipeline.maxTotalThreadsPerThreadgroup();
        let threadgroup_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };
        let num_elements = self.dst.len();
        let grid_size = MTLSize {
            width: num_elements,
            height: 1,
            depth: 1,
        };
        dispatch_threads(&encoder, grid_size, threadgroup_size);
        Ok(())
    }
}

/// A light wrapper around a Metal command buffer that provides a
/// simple API to record high-level operations.
#[derive(Clone)]
pub struct CommandBuffer {
    inner: Rc<CommandBufferInner>,
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
