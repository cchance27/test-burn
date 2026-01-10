#![cfg(test)]
use crate::{
    Context, MetalError, Tensor, kernels::scaled_dot_product_attention::cache::{CacheableSdpa, SdpaKey}
};

mod batch_record_test;
#[cfg(test)]
mod batched_pipeline_parity;
#[cfg(test)]
mod bench_qwen25_matmul;
#[cfg(test)]
mod cacheable_test;
#[cfg(test)]
mod clamping_extreme_test;
#[cfg(test)]
mod determinism_test;
#[cfg(test)]
mod embedding_q8_smoke;
#[cfg(test)]
mod error_path_test;
#[cfg(test)]
mod forward_pass_correctness_test;
#[cfg(test)]
mod fp16_fused_parity;
#[cfg(test)]
mod fp16_transposed_parity;
#[cfg(test)]
mod fused_ffn_swiglu_rmsnorm_parity;
#[cfg(test)]
mod fused_qkv_batched_rmsnorm_parity;
#[cfg(test)]
mod gemv_canonical_batched_parity;
#[cfg(test)]
mod gemv_v2_sdpa_shapes;
#[cfg(test)]
mod generation_test;
#[cfg(test)]
mod gpu_profiling;
#[cfg(test)]
mod kv_cache_write_repeat_kv_heads;
#[cfg(test)]
mod matmul;
#[cfg(test)]
mod matmul_alpha_beta_parity_test;
#[cfg(test)]
mod matmul_backend_regression_test;
#[cfg(test)]
mod matmul_q8_m1_mlx_heuristic_test;
#[cfg(test)]
mod q8_0_quantized_tensor;
#[cfg(test)]
mod q8_parity_qwen25;
#[cfg(test)]
mod q8_proxy_smoke;
#[cfg(test)]
mod q8_qkv_fused;
#[cfg(test)]
mod resource_cache_persistence_test;
#[cfg(test)]
mod rmsnorm_gemv_fused;
#[cfg(test)]
mod sdpa_gemm_batch_parity;
#[cfg(test)]
mod sdpa_materialized_parity;
#[cfg(test)]
mod sdpa_materialized_prefill_parity;
#[cfg(test)]
mod softmax_kernel_consistency_test;
#[cfg(test)]
mod tensor_test;
