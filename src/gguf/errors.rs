use thiserror::Error;

#[derive(Debug, Error)]
pub enum GGUFError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid data")]
    InvalidData,
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
    #[error("Memory mapping error: {0}")]
    MemoryMappingError(String),
    #[error("Dequantization error: {0}")]
    DequantizationError(String),
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
