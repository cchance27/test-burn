#![allow(unused_imports)]
pub use context::{Context, ContextOptions};
pub use error::MetalError;
pub use tensor::{
    BF16Element, Dtype, F16Element, F32Element, KernelElement, KernelTensor, KernelTensorGuard, Tensor as GenericTensor, TensorBF16,
    TensorElement, TensorF16, TensorF32, TensorInit as GenericTensorInit, TensorStorage,
};

pub type Tensor = TensorF32;
pub type TensorInit<'data> = tensor::TensorInit<'data, F32Element>;
pub use tokenizer::{SpecialTokens, Tokenizer, TokenizerError};

pub mod cache_keys;
pub mod cacheable;
pub mod cacheable_resources;
pub mod cacheable_sdpa;
pub mod context;
pub mod encoder;
pub mod error;
pub mod instrumentation;
pub mod metrics;
pub mod operation;
pub mod pool;
pub mod resource_cache;
pub mod tensor;
pub mod tokenizer;
pub use operation::{CommandBuffer, Operation};

pub mod generation;
pub mod models;

pub mod kernels;

mod tests;
