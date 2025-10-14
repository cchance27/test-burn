#![cfg(test)]
use crate::cache_keys::SdpaKey;
use crate::cacheable::Cacheable as _;
use crate::cacheable_sdpa::CacheableSdpa;

use crate::{Context, MetalError, Tensor};

mod cacheable_test;
mod clamping_extreme_test;
mod determinism_test;
mod error_path_test;
mod forward_pass_correctness_test;
mod generation_test;
mod gpu_profiling;
mod matmul;
mod matmul_alpha_beta_parity_test;
mod matmul_backend_regression_test;
mod matmul_dispatch_test;
mod resource_cache_persistence_test;
mod softmax_kernel_consistency_test;
mod tensor_test;
