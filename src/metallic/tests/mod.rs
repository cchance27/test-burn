#![cfg(test)]
use crate::metallic::cache_keys::SdpaKey;
use crate::metallic::cacheable::Cacheable as _;
use crate::metallic::cacheable_sdpa::CacheableSdpa;

use crate::metallic::kernels::softmax::SoftmaxOp;
use crate::metallic::{Context, MetalError, Tensor};

mod cacheable_test;
mod clamping_extreme_test;
mod determinism_test;
mod error_path_test;
mod forward_pass_correctness_test;
mod generation_test;
mod tensor_test;
