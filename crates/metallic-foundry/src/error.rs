use thiserror::Error;

use crate::tensor::Dtype;

#[derive(Error, Debug)]
pub enum GemvError {
    #[error("Invalid GEMV vector shape: {actual:?}, expected (1, K)")]
    VectorShape { actual: Vec<usize> },
    #[error("Invalid GEMV matrix shape: {actual:?}, expected (K={expected_k}, N)")]
    MatrixShape { expected_k: usize, actual: Vec<usize> },
    #[error("Bias length {actual} must equal N {expected}")]
    BiasLengthMismatch { expected: usize, actual: usize },
    #[error("Residual shape {actual:?} must be [1, N={expected}]")]
    ResidualShapeMismatch { expected: usize, actual: Vec<usize> },
    #[error("Q8 GEMV expects one dimension equal to K ({expected_k}), got {actual:?}")]
    QuantShape { expected_k: usize, actual: Vec<usize> },
    #[error("Q8 canonical weights_per_block cannot be zero")]
    InvalidQuantParams,
}

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
    #[error("Load library failed: {0}")]
    LoadLibraryFailed(String),
    #[error("Function creation failed: {0}")]
    FunctionCreationFailed(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineCreationFailed(String),
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
    #[error("Operation failed: {0}")]
    OperationFailed(String),
    #[error("Gemv error: {0}")]
    Gemv(#[from] GemvError),
    #[error("Input not found: {0}")]
    InputNotFound(String),
    #[error("Pooled tensor used after pool reset (Use-After-Free)")]
    UseAfterFree,
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
