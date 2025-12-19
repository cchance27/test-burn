#![cfg(test)]
use crate::{
    Context, MetalError, Tensor, kernels::scaled_dot_product_attention::cache::{CacheableSdpa, SdpaKey}
};

mod batch_record_test;
mod bench_qwen25_matmul;
mod cacheable_test;
mod clamping_extreme_test;
mod determinism_test;
mod error_path_test;
mod forward_pass_correctness_test;
mod fp16_transposed_parity;
mod generation_test;
mod gpu_profiling;
mod matmul;
mod matmul_alpha_beta_parity_test;
mod matmul_backend_regression_test;
mod matmul_dispatch_test;
mod matmul_q8_m1_mlx_heuristic_test;
mod q8_0_quantized_tensor;
mod q8_parity_qwen25;
mod q8_proxy_smoke;
mod q8_qkv_fused;
mod resource_cache_persistence_test;
mod rmsnorm_gemv_fused;
mod softmax_kernel_consistency_test;
mod tensor_test;
