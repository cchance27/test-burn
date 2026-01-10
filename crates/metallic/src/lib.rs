extern crate self as metallic;

pub use context::{Context, SamplerBuffers};
pub use error::{GemvError, MetalError};
pub use operation::{CommandBuffer, ComputeKernelEncoder, Operation};
pub use tensor::{
    Dtype, F16Element, F32Element, Q8_0_BLOCK_SIZE_BYTES, Q8_0_WEIGHTS_PER_BLOCK, QuantizedQ8_0Tensor, Tensor, TensorElement, TensorInit, TensorStorage
};
pub use tokenizer::{SpecialTokens, Tokenizer, TokenizerError};

pub mod alternatives;
pub mod caching;
pub mod context;
pub mod encoder;
pub mod error;
pub mod generation;
pub mod gguf;

pub mod models;
pub mod operation;
pub mod pool;
pub mod profiling_state;
pub mod tensor;
pub mod tokenizer;

pub mod macros;

pub mod foundry;
pub mod metals;
pub mod policies;
pub mod safety;
pub mod tests;
pub mod types;

pub mod kernels;

pub mod compound;

pub mod fusion;

pub use caching::{CacheMetricsSnapshot, CacheStats, ResourceCache, TensorPreparationCache, TensorPreparationMetrics};

#[cfg(all(feature = "src_kernels", feature = "built_kernels"))]
compile_error!("features `src_kernels` and `built_kernels` are mutually exclusive — pick only one.");
