use objc2::{rc::Retained, runtime::ProtocolObject};
pub use objc2_metal::{MTLBarrierScope as MetalBarrierScope, MTLResourceOptions as MetalResourceOptions};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary
};

pub mod dispatch;
pub mod metal;
pub mod semantic;
pub mod storage;
pub mod stream;

pub use dispatch::{DispatchConfig, GridSize, ThreadgroupSize};
pub use stream::{CommandStream, StreamEncoder};

use crate::MetalError;

// Thread-safe Metal wrappers.
// Thread-safe Metal wrappers.
#[derive(Clone, Debug)]
pub struct MetalDevice(pub Retained<ProtocolObject<dyn MTLDevice>>);

#[derive(Clone, Debug)]
pub struct MetalQueue(pub Retained<ProtocolObject<dyn MTLCommandQueue>>);

#[derive(Clone, Debug)]
pub struct MetalBuffer(pub Retained<ProtocolObject<dyn MTLBuffer>>);

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}
unsafe impl Send for MetalQueue {}
unsafe impl Sync for MetalQueue {}
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

// Deref to underlying Metal object for backward compatibility/ease of use
impl std::ops::Deref for MetalDevice {
    type Target = ProtocolObject<dyn MTLDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalDevice {
    /// Get the unique registry ID for this device.
    ///
    /// Used for pipeline cache keys to support multi-GPU scenarios.
    pub fn registry_id(&self) -> u64 {
        use objc2_metal::MTLDevice as _;
        self.0.registryID()
    }

    pub fn new_buffer(&self, length: usize, options: MetalResourceOptions) -> Option<MetalBuffer> {
        self.0.newBufferWithLength_options(length, options).map(MetalBuffer)
    }

    pub fn new_buffer_with_bytes(
        &self,
        bytes: std::ptr::NonNull<std::ffi::c_void>,
        length: usize,
        options: MetalResourceOptions,
    ) -> Option<MetalBuffer> {
        unsafe { self.0.newBufferWithBytes_length_options(bytes, length, options).map(MetalBuffer) }
    }
    pub fn new_buffer_with_bytes_no_copy(
        &self,
        bytes: std::ptr::NonNull<std::ffi::c_void>,
        length: usize,
        options: MetalResourceOptions,
        deallocator: Option<&block2::Block<dyn Fn(std::ptr::NonNull<std::ffi::c_void>, usize)>>,
    ) -> Option<MetalBuffer> {
        unsafe {
            self.0
                .newBufferWithBytesNoCopy_length_options_deallocator(bytes, length, options, deallocator)
                .map(MetalBuffer)
        }
    }

    pub fn new_buffer_from_slice<T: Copy>(&self, data: &[T], options: MetalResourceOptions) -> Option<MetalBuffer> {
        let size = std::mem::size_of_val(data);
        if size == 0 {
            return None;
        }
        let ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void)?;
        unsafe { self.0.newBufferWithBytes_length_options(ptr, size, options).map(MetalBuffer) }
    }
    pub fn create_system_default_device() -> Result<MetalDevice, crate::error::MetalError> {
        objc2_metal::MTLCreateSystemDefaultDevice()
            .map(MetalDevice)
            .ok_or(crate::error::MetalError::DeviceNotFound)
    }

    pub fn recommended_max_working_set_size(&self) -> u64 {
        use objc2_metal::MTLDevice as _;
        self.0.recommendedMaxWorkingSetSize()
    }

    pub fn new_command_queue(&self) -> Result<MetalQueue, crate::error::MetalError> {
        use objc2_metal::MTLDevice as _;
        self.0
            .newCommandQueue()
            .map(MetalQueue)
            .ok_or(crate::error::MetalError::CommandQueueCreationFailed)
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MetalCompileOptions>,
    ) -> Result<MetalLibrary, crate::error::MetalError> {
        use objc2_metal::MTLDevice as _;
        let ns_source = objc2_foundation::NSString::from_str(source);
        let options = options.map(|o| &*o.0); // deref stored retained pointer
        self.0
            .newLibraryWithSource_options_error(&ns_source, options)
            .map(MetalLibrary)
            .map_err(|e| crate::error::MetalError::LoadLibraryFailed(format!("{:?}", e)))
    }

    pub fn new_default_library(&self) -> Option<MetalLibrary> {
        use objc2_metal::MTLDevice as _;
        self.0.newDefaultLibrary().map(MetalLibrary)
    }

    pub fn new_compute_pipeline_state_with_function(&self, function: &MetalFunction) -> Result<MetalPipeline, crate::error::MetalError> {
        use objc2_metal::MTLDevice as _;
        // map error to string properly
        self.0
            .newComputePipelineStateWithFunction_error(&function.0)
            .map(MetalPipeline)
            .map_err(|e| crate::error::MetalError::PipelineCreationFailed(format!("{:?}", e)))
    }
}

impl std::ops::Deref for MetalQueue {
    type Target = ProtocolObject<dyn MTLCommandQueue>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalQueue {
    pub fn command_buffer(&self) -> Result<MetalCommandBuffer, crate::error::MetalError> {
        self.0
            .commandBuffer()
            .map(MetalCommandBuffer)
            .ok_or(crate::error::MetalError::CommandBufferCreationFailed)
    }
}

impl std::ops::Deref for MetalBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalBuffer {
    /// Create a MetalBuffer from a raw Retained MTLBuffer.
    pub fn from_retained(buffer: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Self(buffer)
    }

    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.0.contents().as_ptr()
    }

    pub fn length(&self) -> usize {
        use objc2_metal::MTLBuffer as _;
        self.0.length()
    }

    pub fn as_ptr_addr(&self) -> usize {
        objc2::rc::Retained::as_ptr(&self.0) as usize
    }

    /// Read buffer contents into a Vec<T>.
    /// # Safety
    /// The buffer must contain valid data for T, and T must be Plain Old Data.
    pub fn read_to_vec<T: Clone>(&self, count: usize) -> Vec<T> {
        unsafe {
            let ptr = self.contents() as *const T;
            std::slice::from_raw_parts(ptr, count).to_vec()
        }
    }

    /// Read a scalar value from the buffer.
    /// # Safety
    /// The buffer must be large enough and aligned for T.
    pub fn read_scalar<T: Copy>(&self) -> T {
        unsafe {
            let ptr = self.contents() as *const T;
            std::ptr::read_volatile(ptr)
        }
    }

    /// Fill a range of the buffer with a byte value.
    pub fn fill_bytes(&self, value: u8, count: usize) {
        unsafe {
            let ptr = self.contents() as *mut u8;
            std::ptr::write_bytes(ptr, value, count);
        }
    }

    /// Copy a slice of data into the buffer.
    /// # Safety
    /// T must be Plain Old Data. The buffer must be large enough.
    pub fn copy_from_slice<T: Copy>(&self, data: &[T]) {
        unsafe {
            let ptr = self.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Copy a slice of data into the buffer at a specific element offset.
    /// # Safety
    /// T must be Plain Old Data. The buffer must be large enough.
    pub fn copy_from_slice_offset<T: Copy>(&self, data: &[T], offset_elements: usize) {
        unsafe {
            let ptr = self.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset_elements), data.len());
        }
    }

    /// Create a mutable slice of the buffer contents.
    /// # Safety
    /// The caller must ensure exclusive access if mutating.
    pub unsafe fn as_slice_mut<T>(&self, count: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.contents() as *mut T, count) }
    }

    /// Access buffer contents as a mutable slice within a closure.
    /// This ensures the slice lifetime is bounded by the closure, preventing use-after-free logic errors slightly better than returning a slice.
    /// # Safety
    /// Caller must ensure exclusive access (no other threads reading/writing/uploading to this buffer).
    pub fn write_via_slice<T: Copy, F: FnOnce(&mut [T])>(&self, count: usize, f: F) {
        unsafe {
            let ptr = self.contents() as *mut T;
            let slice = std::slice::from_raw_parts_mut(ptr, count);
            f(slice);
        }
    }
}

/// Cast a reference safely if TypeIds match.
pub fn cast_from_ref<Src: 'static, Dst: 'static + Copy>(src: &Src) -> Option<Dst> {
    if std::any::TypeId::of::<Src>() == std::any::TypeId::of::<Dst>() {
        // Safety: TypeId match guarantees layout compatibility for these primitive types
        unsafe { Some(std::ptr::read(src as *const Src as *const Dst)) }
    } else {
        None
    }
}

/// View a POD slice as bytes.
pub fn pod_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

pub type Device = MetalDevice;
pub type Queue = MetalQueue;
pub type Buffer = MetalBuffer;

use crate::tensor::Dtype;

/// Protocol for kernel buffer arguments.
pub trait KernelArg {
    /// Get the underlying Metal buffer.
    fn buffer(&self) -> &Buffer;

    /// Get the byte offset into the buffer.
    fn offset(&self) -> usize;

    /// Get the data type of the underlying tensor.
    fn dtype(&self) -> Dtype;

    /// Get the tensor dimensions.
    fn dims(&self) -> &[usize] {
        &[]
    }

    /// Get the tensor strides.
    fn strides(&self) -> &[usize] {
        &[]
    }

    /// Flush host writes to the GPU.
    fn flush(&self) {}
}

impl<T: KernelArg + ?Sized> KernelArg for &T {
    fn buffer(&self) -> &Buffer {
        (**self).buffer()
    }

    fn offset(&self) -> usize {
        (**self).offset()
    }

    fn dtype(&self) -> Dtype {
        (**self).dtype()
    }

    fn dims(&self) -> &[usize] {
        (**self).dims()
    }

    fn strides(&self) -> &[usize] {
        (**self).strides()
    }

    fn flush(&self) {
        (**self).flush()
    }
}

use smallvec::SmallVec;

/// A kernel argument that captures buffer+offset from any Tensor.
#[derive(Clone, Debug)]
pub struct TensorArg {
    pub buffer: Option<Buffer>,
    pub offset: usize,
    pub dtype: Dtype,
    pub dims: SmallVec<[usize; 4]>,
    pub strides: SmallVec<[usize; 4]>,
}

impl Default for TensorArg {
    fn default() -> Self {
        Self {
            buffer: None,
            offset: 0,
            dtype: Dtype::F16,
            dims: SmallVec::new(),
            strides: SmallVec::new(),
        }
    }
}

unsafe impl Send for TensorArg {}
unsafe impl Sync for TensorArg {}

impl TensorArg {
    /// Create a TensorArg from any type implementing KernelArg.
    pub fn from_tensor<K: KernelArg>(arg: &K) -> Self {
        arg.flush();
        Self {
            buffer: Some(arg.buffer().clone()),
            offset: arg.offset(),
            dtype: arg.dtype(),
            dims: SmallVec::from_slice(arg.dims()),
            strides: SmallVec::from_slice(arg.strides()),
        }
    }

    /// Create a TensorArg from a raw Metal buffer.
    pub fn from_buffer(buffer: Buffer, dtype: Dtype, dims: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            buffer: Some(buffer),
            offset: 0,
            dtype,
            dims: SmallVec::from_vec(dims),
            strides: SmallVec::from_vec(strides),
        }
    }

    /// Create a dummy TensorArg for testing.
    pub fn dummy(buffer: Buffer) -> Self {
        Self {
            buffer: Some(buffer),
            offset: 0,
            dtype: Dtype::F16,
            dims: SmallVec::new(),
            strides: SmallVec::new(),
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl KernelArg for TensorArg {
    fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().expect("Attempted to access buffer on null TensorArg")
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn dims(&self) -> &[usize] {
        &self.dims
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }
}

/// A compiled Compute Pipeline State.
#[derive(Clone, Debug)]
pub struct MetalPipeline(pub(crate) Retained<ProtocolObject<dyn MTLComputePipelineState>>);

unsafe impl Send for MetalPipeline {}
unsafe impl Sync for MetalPipeline {}

impl std::ops::Deref for MetalPipeline {
    type Target = ProtocolObject<dyn MTLComputePipelineState>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type Pipeline = MetalPipeline;

/// A compiled Metal Library.
#[derive(Clone, Debug)]
pub struct MetalLibrary(pub(crate) Retained<ProtocolObject<dyn MTLLibrary>>);

unsafe impl Send for MetalLibrary {}
unsafe impl Sync for MetalLibrary {}

impl std::ops::Deref for MetalLibrary {
    type Target = ProtocolObject<dyn MTLLibrary>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalLibrary {
    pub fn new_function(&self, name: &str) -> Option<MetalFunction> {
        use objc2_metal::MTLLibrary as _;
        let ns_name = objc2_foundation::NSString::from_str(name);
        self.0.newFunctionWithName(&ns_name).map(MetalFunction)
    }
}

pub type Library = MetalLibrary;

/// A function within a Metal Library.
#[derive(Clone, Debug)]
pub struct MetalFunction(pub(crate) Retained<ProtocolObject<dyn MTLFunction>>);

unsafe impl Send for MetalFunction {}
unsafe impl Sync for MetalFunction {}

impl std::ops::Deref for MetalFunction {
    type Target = ProtocolObject<dyn MTLFunction>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type Function = MetalFunction;

/// Wrapper for Metal Compile Options
#[derive(Clone, Debug)]
pub struct MetalCompileOptions(pub(crate) Retained<objc2_metal::MTLCompileOptions>);

impl Default for MetalCompileOptions {
    fn default() -> Self {
        let options = objc2_metal::MTLCompileOptions::new();
        options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);
        options.setEnableLogging(true);
        Self(options)
    }
}

impl MetalCompileOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A compute command encoder.
#[derive(Clone, Debug)]
pub struct ComputeCommandEncoder(pub(crate) Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>);

unsafe impl Send for ComputeCommandEncoder {}
unsafe impl Sync for ComputeCommandEncoder {}

impl std::ops::Deref for ComputeCommandEncoder {
    type Target = ProtocolObject<dyn MTLComputeCommandEncoder>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ComputeCommandEncoder {
    pub fn set_compute_pipeline_state(&self, pipeline: &MetalPipeline) {
        self.0.setComputePipelineState(pipeline);
    }

    /// Set a buffer at the given index with an offset.
    pub fn set_buffer(&self, index: u32, buffer: &MetalBuffer, offset: usize) {
        self.set_buffer_opt(index, Some(buffer), offset);
    }

    /// Set a buffer at the given index with an offset, allowing for None.
    pub fn set_buffer_opt(&self, index: u32, buffer: Option<&MetalBuffer>, offset: usize) {
        use objc2_metal::MTLComputeCommandEncoder as _;
        unsafe {
            self.0.setBuffer_offset_atIndex(buffer.map(|b| &**b), offset, index as usize);
        }
    }

    /// Set a buffer at the given index from a TensorArg.
    pub fn set_tensor_arg(&self, index: u32, arg: &crate::types::TensorArg) {
        if let Some(buffer) = &arg.buffer {
            self.set_buffer(index, buffer, arg.offset);
        }
    }

    /// Set a small amount of literal data at the given index.
    pub fn set_bytes<T>(&self, index: u32, data: &T) {
        use objc2_metal::MTLComputeCommandEncoder as _;
        let ptr = std::ptr::NonNull::from(data).cast();
        unsafe {
            self.0.setBytes_length_atIndex(ptr, std::mem::size_of::<T>(), index as usize);
        }
    }

    pub fn dispatch_threadgroups(&self, grid_size: crate::types::dispatch::GridSize, group_size: crate::types::dispatch::ThreadgroupSize) {
        let (grid, group) = (grid_size.into(), group_size.into());
        self.0.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
    }

    /// Dispatch threadgroups without inserting a memory barrier.
    ///
    /// Use this only when you know the dispatched kernel has no RAW/WAR/WAW hazards with
    /// subsequent encodes on the same command encoder, or when synchronization is handled
    /// elsewhere (e.g. explicit barriers or separate command buffers).
    #[inline]
    pub fn dispatch_unbarriered(
        &self,
        grid_size: crate::types::dispatch::GridSize,
        group_size: crate::types::dispatch::ThreadgroupSize,
    ) {
        self.dispatch_threadgroups(grid_size, group_size);
    }

    /// Dispatch threadgroups and immediately apply a buffer memory barrier.
    pub fn dispatch_with_barrier(&self, grid_size: crate::types::dispatch::GridSize, group_size: crate::types::dispatch::ThreadgroupSize) {
        self.dispatch_threadgroups(grid_size, group_size);
        self.memory_barrier(MetalBarrierScope::Buffers);
    }

    /// Convenience helper to set pipeline, bind arguments, and dispatch.
    pub fn execute<F>(
        &self,
        pipeline: &MetalPipeline,
        grid: crate::types::dispatch::GridSize,
        group: crate::types::dispatch::ThreadgroupSize,
        binder: F,
    ) where
        F: FnOnce(&Self),
    {
        self.set_compute_pipeline_state(pipeline);
        binder(self);
        self.dispatch_with_barrier(grid, group);
    }

    /// Convenience helper to set pipeline, bind arguments, and dispatch without a memory barrier.
    #[inline]
    pub fn execute_unbarriered<F>(
        &self,
        pipeline: &MetalPipeline,
        grid: crate::types::dispatch::GridSize,
        group: crate::types::dispatch::ThreadgroupSize,
        binder: F,
    ) where
        F: FnOnce(&Self),
    {
        self.set_compute_pipeline_state(pipeline);
        binder(self);
        self.dispatch_unbarriered(grid, group);
    }

    pub fn end_encoding(&self) {
        self.0.endEncoding();
    }

    pub fn memory_barrier(&self, scope: objc2_metal::MTLBarrierScope) {
        self.0.memoryBarrierWithScope(scope);
    }
}

/// A blit command encoder.
#[derive(Clone, Debug)]
pub struct BlitCommandEncoder(pub(crate) Retained<ProtocolObject<dyn objc2_metal::MTLBlitCommandEncoder>>);

unsafe impl Send for BlitCommandEncoder {}
unsafe impl Sync for BlitCommandEncoder {}

impl std::ops::Deref for BlitCommandEncoder {
    type Target = ProtocolObject<dyn objc2_metal::MTLBlitCommandEncoder>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl BlitCommandEncoder {
    pub fn copy_from_buffer(
        &self,
        source_buffer: &MetalBuffer,
        source_offset: usize,
        destination_buffer: &MetalBuffer,
        destination_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.0.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                source_buffer,
                source_offset,
                destination_buffer,
                destination_offset,
                size,
            );
        }
    }

    pub fn end_encoding(&self) {
        self.0.endEncoding();
    }
}

/// A command buffer.
#[derive(Clone, Debug)]
pub struct MetalCommandBuffer(pub(crate) Retained<ProtocolObject<dyn MTLCommandBuffer>>);

unsafe impl Send for MetalCommandBuffer {}
unsafe impl Sync for MetalCommandBuffer {}

impl std::ops::Deref for MetalCommandBuffer {
    type Target = ProtocolObject<dyn MTLCommandBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalCommandBuffer {
    pub fn compute_command_encoder(&self) -> Result<ComputeCommandEncoder, MetalError> {
        self.0
            .computeCommandEncoder()
            .map(ComputeCommandEncoder)
            .ok_or(MetalError::CommandQueueCreationFailed)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder, MetalError> {
        self.0
            .blitCommandEncoder()
            .map(BlitCommandEncoder)
            .ok_or(MetalError::CommandQueueCreationFailed)
    }

    pub fn commit(&self) {
        self.0.commit();
    }

    pub fn wait_until_completed(&self) {
        self.0.waitUntilCompleted();
    }

    pub fn gpu_start_time(&self) -> f64 {
        use objc2_metal::MTLCommandBuffer as _;
        self.0.GPUStartTime()
    }

    pub fn gpu_end_time(&self) -> f64 {
        use objc2_metal::MTLCommandBuffer as _;
        self.0.GPUEndTime()
    }

    pub fn add_completed_handler<F>(&self, handler: F)
    where
        F: Fn(&MetalCommandBuffer) + Send + Sync + 'static,
    {
        use std::{
            ptr::NonNull, sync::{Arc, Mutex}
        };

        use block2::RcBlock;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::MTLCommandBuffer as _;

        let handler = Arc::new(Mutex::new(Some(handler)));
        let block = RcBlock::new(move |cmd: NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
            let mut guard = handler.lock().unwrap();
            if let Some(h) = guard.take() {
                let retained = unsafe { objc2::rc::Retained::retain(cmd.as_ptr()) }.expect("Command buffer should be valid");
                let wrapper = MetalCommandBuffer(retained);
                h(&wrapper);
            }
        });

        let raw_block = RcBlock::into_raw(block);
        unsafe {
            self.0.addCompletedHandler(raw_block.cast());
        }
    }
}

pub type CommandBuffer = MetalCommandBuffer;
