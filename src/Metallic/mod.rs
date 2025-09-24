#![allow(unused_imports)]
pub use context::Context;
pub use error::MetalError;
pub use matmul::{encode_mps_matrix_multiplication, mps_matrix_from_buffer};
pub use softmax::ensure_fused_softmax_pipeline;
pub use tensor::Tensor;
pub use tokenizer::{SpecialTokens, Tokenizer, TokenizerError};

pub mod cache_keys;
pub mod cacheable;
pub mod cacheable_resources;
pub mod cacheable_sdpa;
pub mod context;
pub mod encoder;
pub mod error;
pub mod matmul;
pub mod operation;
pub mod pool;
pub mod resource_cache;
pub mod scaled_dot_product_attention;
pub mod softmax;
pub mod tensor;
pub mod tokenizer;

pub use operation::{CommandBuffer, Operation};

pub mod elemwise_add;
pub mod elemwise_div;
pub mod elemwise_mul;
pub mod elemwise_sub;
pub mod gelu;
pub mod generation;
pub mod kv_rearrange;
pub mod layernorm;
pub mod model;
pub mod permute;
pub mod qwen25;
pub mod rmsnorm;
pub mod rope;
pub mod silu;
pub mod swiglu;

#[cfg(test)]
mod tests;
