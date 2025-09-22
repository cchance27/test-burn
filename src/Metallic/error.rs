use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetalError {
    #[error("Device not found")]
    DeviceNotFound,
    #[error("Command queue creation failed")]
    CommandQueueCreationFailed,
    #[error("Command buffer creation failed")]
    CommandBufferCreationFailed,
    #[error("Compute encoder creation failed")]
    ComputeEncoderCreationFailed,
    #[error("Buffer creation failed with size {0}")]
    BufferCreationFailed(usize),
    #[error("Buffer from bytes creation failed")]
    BufferFromBytesCreationFailed,
    #[error("Null pointer")]
    NullPointer,
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Library compilation failed: {0}")]
    LibraryCompilationFailed(String),
    #[error("Function creation failed: {0}")]
    FunctionCreationFailed(String),
    #[error("Pipeline creation failed")]
    PipelineCreationFailed,
    #[error("Resource not cached: {0}")]
    ResourceNotCached(String),
    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}
