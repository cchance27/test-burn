#![allow(unused_imports)]
pub use context::Context;
pub use error::MetalError;
pub use matmul::{encode_mps_matrix_multiplication, mps_matrix_from_buffer};

// TODO: We added an ecode_fused_softmax, but we didn't use it ... not sure what this was for.
pub use softmax::ensure_fused_softmax_pipeline;
pub use tensor::Tensor;

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

pub use operation::{CommandBuffer, Operation};
