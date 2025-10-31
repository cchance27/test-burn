pub use context::{Context, SamplerBuffers};
pub use error::MetalError;
pub use operation::{CommandBuffer, ComputeKernelEncoder, Operation};
pub use tensor::{Dtype, F16Element, F32Element, Tensor, TensorElement, TensorInit, TensorStorage};
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
pub mod softmax_utils;
pub mod tensor;
pub mod tokenizer;

pub mod mps_graph;

pub mod macros;

pub mod kernels;

mod tests;

pub use caching::{CacheMetricsSnapshot, CacheStats, ResourceCache, TensorPreparationCache, TensorPreparationMetrics};

#[cfg(all(feature = "src_kernels", feature = "built_kernels"))]
compile_error!("features `src_kernels` and `built_kernels` are mutually exclusive — pick only one.");
