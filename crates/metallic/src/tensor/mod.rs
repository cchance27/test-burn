use std::{
    cell::RefCell, marker::PhantomData, rc::Rc, sync::{Arc, Mutex}
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandQueue, MTLDevice};

use super::{Context, MetalError, operation::CommandBuffer};

pub type RetainedBuffer = Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>;

pub mod dtypes;
pub mod enums;
pub mod generation;
pub mod host_access;
pub mod host_access_methods;
pub mod mps_views;
pub mod operations;
pub mod quantized;
pub mod storage;
pub mod utility_methods;

pub use dtypes::*;
pub use enums::{TensorInit, TensorStorage};
use host_access::DEFAULT_CPU_FILL_THRESHOLD_MB;
pub use host_access::{
    HostAccessRegistry, HostAccessState, ThreadSafeBuffer, buffer_registry_key, host_access_registry, shared_host_access_state
};
pub use mps_views::MpsMatrixBatchView;
pub use quantized::{Q8_0_BLOCK_SIZE_BYTES, Q8_0_WEIGHTS_PER_BLOCK, Q8_0Block, QuantizedQ8_0Tensor};

const TENSOR_STAGING_READ_OP: &str = "tensor_staging_read";
const TENSOR_STAGING_PREP_OP: &str = "tensor_staging_prepare";
const TENSOR_STAGING_FLUSH_OP: &str = "tensor_staging_flush";

#[derive(Clone)]
pub struct Tensor<T: TensorElement> {
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
    host_accessible: bool,
    host_access: Arc<Mutex<HostAccessState>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    /// The command buffer that must complete before this tensor's data is safe for host access.
    /// None indicates the tensor is already synchronized with the CPU.
    pub(crate) defining_cmd_buffer: Rc<RefCell<Option<CommandBuffer>>>,
    marker: PhantomData<T>,
}

// Note: Implementation methods are in separate modules: storage, host_access_methods,
// utility_methods, operations, and generation to keep this file focused and small

// Re-export common aliases for compatibility
pub type F32Element = F32;
pub type F16Element = F16;

// Lightweight unions to let the matmul dispatcher accept either dense or quantized RHS.
// Start with Q8_0 only; extendable to more quant formats later.
pub enum QuantizedTensor<'a> {
    Q8_0(&'a QuantizedQ8_0Tensor),
}

pub enum TensorType<'a, T: TensorElement> {
    Dense(&'a Tensor<T>),
    Quant(QuantizedTensor<'a>),
}
