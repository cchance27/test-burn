#![cfg(test)]
use crate::{
    Context, MetalError, Tensor, kernels::scaled_dot_product_attention::cache::{CacheableSdpa, SdpaKey}
};

mod batch_record_test;

#[cfg(test)]
mod cacheable_test;
#[cfg(test)]
mod clamping_extreme_test;
#[cfg(test)]
mod determinism_test;

#[cfg(test)]
mod error_path_test;
#[cfg(test)]
mod forward_pass_correctness_test;

#[cfg(test)]
mod generation_test;

#[cfg(test)]
mod matmul_backend_regression_test;
#[cfg(test)]
mod q8_0_quantized_tensor;
#[cfg(test)]
mod resource_cache_persistence_test;

#[cfg(test)]
mod tensor_test;
