use thiserror::Error;

use crate::tensor::Dtype;

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
    #[error("Unsupported dtype {dtype:?} for {operation}")]
    UnsupportedDtype { operation: &'static str, dtype: Dtype },
    #[error("Resource cache is required but not provided")]
    ResourceCacheRequired,
    #[error("Tokenizer error: {0}")]
    TokenizerError(Box<crate::tokenizer::TokenizerError>),
    #[error("Regex Error: {0}")]
    RegexError(Box<fancy_regex::Error>),
    #[error("Tensor dtype mismatch: expected {expected:?}, got {actual:?}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },
}

impl From<crate::tokenizer::TokenizerError> for MetalError {
    fn from(e: crate::tokenizer::TokenizerError) -> Self {
        MetalError::TokenizerError(Box::new(e))
    }
}

impl From<fancy_regex::Error> for MetalError {
    fn from(e: fancy_regex::Error) -> Self {
        MetalError::RegexError(Box::new(e))
    }
}
