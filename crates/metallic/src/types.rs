use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};

pub mod dispatch;
pub mod metal;
pub mod semantic;
pub mod storage;

pub use dispatch::{DispatchConfig, GridSize, ThreadgroupSize};

// Thread-safe Metal wrappers.
// Thread-safe Metal wrappers.
#[derive(Clone, Debug)]
pub struct MetalDevice(pub(crate) Retained<ProtocolObject<dyn MTLDevice>>);

#[derive(Clone, Debug)]
pub struct MetalQueue(pub(crate) Retained<ProtocolObject<dyn MTLCommandQueue>>);

#[derive(Clone, Debug)]
pub struct MetalBuffer(pub(crate) Retained<ProtocolObject<dyn MTLBuffer>>);

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

impl std::ops::Deref for MetalQueue {
    type Target = ProtocolObject<dyn MTLCommandQueue>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Deref for MetalBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
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

    fn flush(&self) {
        (**self).flush()
    }
}

/// A kernel argument that captures buffer+offset from any Tensor.
#[derive(Clone)]
pub struct TensorArg {
    buffer: Option<Buffer>,
    offset: usize,
    dtype: Dtype,
}

impl Default for TensorArg {
    fn default() -> Self {
        Self {
            buffer: None,
            offset: 0,
            dtype: Dtype::F16,
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
        }
    }

    /// Create a dummy TensorArg for testing.
    pub fn dummy(buffer: Buffer) -> Self {
        Self {
            buffer: Some(buffer),
            offset: 0,
            dtype: Dtype::F16,
        }
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

/// A compute command encoder.
#[derive(Clone, Debug)]
pub struct ComputeCommandEncoder(pub(crate) Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>);

unsafe impl Send for ComputeCommandEncoder {}
unsafe impl Sync for ComputeCommandEncoder {}

impl std::ops::Deref for ComputeCommandEncoder {
    type Target = ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
