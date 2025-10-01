use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Sets the compute pipeline state for a command encoder.
#[inline]
pub fn set_compute_pipeline_state(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    state: &ProtocolObject<dyn MTLComputePipelineState>,
) {
    encoder.setComputePipelineState(state);
}

/// Sets a buffer for a compute kernel.
#[inline]
pub fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), offset, index) };
}

/// Sets a small amount of data for a compute kernel.
#[inline]
pub fn set_bytes<T: Sized>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, data: &T) {
    let size = std::mem::size_of::<T>();
    // SAFETY: `data` is a valid reference, so its pointer is non-null.
    // Convert the reference into a NonNull<c_void> as required by the objc2 binding.
    unsafe {
        let ptr = std::ptr::NonNull::from(data).cast::<c_void>();
        encoder.setBytes_length_atIndex(ptr, size, index);
    }
}

/// Sets a slice of data for a compute kernel.
#[inline]
pub fn set_bytes_slice<T: Sized>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, data: &[T]) {
    let size = std::mem::size_of::<T>() * data.len();

    unsafe {
        if size == 0 {
            let ptr = std::ptr::NonNull::<T>::dangling().cast::<c_void>();
            encoder.setBytes_length_atIndex(ptr, 0, index);
            return;
        }

        let ptr = std::ptr::NonNull::from(&data[0]).cast::<c_void>();
        encoder.setBytes_length_atIndex(ptr, size, index);
    }
}

/// Dispatches a compute kernel.
#[inline]
pub fn dispatch_threadgroups(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, groups: MTLSize, threads_per_tg: MTLSize) {
    debug_assert!(groups.width > 0, "groups.width must be non-zero");
    debug_assert!(groups.height > 0, "groups.height must be non-zero");
    debug_assert!(groups.depth > 0, "groups.depth must be non-zero");
    debug_assert!(threads_per_tg.width > 0, "threads_per_tg.width must be non-zero");
    debug_assert!(threads_per_tg.height > 0, "threads_per_tg.height must be non-zero");
    debug_assert!(threads_per_tg.depth > 0, "threads_per_tg.depth must be non-zero");

    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_tg);
}

/// Dispatches a compute kernel using thread-level parallelism.
#[inline]
pub fn dispatch_threads(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, grid_size: MTLSize, threadgroup_size: MTLSize) {
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
}
