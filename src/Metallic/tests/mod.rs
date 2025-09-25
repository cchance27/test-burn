use crate::metallic::Operation as _;
use crate::metallic::cache_keys::SdpaKey;
use crate::metallic::cacheable::Cacheable as _;
use crate::metallic::cacheable_sdpa::CacheableSdpa;

use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
use crate::metallic::ensure_fused_softmax_pipeline;
use crate::metallic::matmul::MatMulOperation;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Tensor};
use objc2_metal::MTLComputePipelineState as _;

use crate::metallic::softmax::SoftmaxOperation;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResourceOptions};
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice};
use std::ffi::c_void;
use std::time::Instant;

use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};

use crate::metallic::Operation as _;
use rand::Rng as _;

mod cacheable_test;
mod clamping_extreme_test;
mod determinism_test;
mod elemwise_add_test;
mod elemwise_div_test;
mod elemwise_mul_test;
mod elemwise_sub_test;
mod error_path_test;
mod forward_pass_correctness_test;
mod gelu_test;
mod generation_test;
mod layernorm_test;
mod matmul_alpha_beta_test;
mod matmul_extreme_test;
mod matmul_offset_test;
mod matmul_test;
mod matmul_transpose_test;
mod model_test;
mod permute_reassembly_test;
mod rmsnorm_test;
mod rope_extreme_test;
mod rope_test;
mod scaled_dot_product_attention_test;
mod sdpa_extreme_test;
mod sdpa_numerical_stability_test;
mod sdpa_property_test;
mod silu_extreme_test;
mod silu_test;
mod softmax_extremes_additional_test;
mod softmax_extremes_test;
mod softmax_irregular_test;
mod softmax_pipeline_test;
mod softmax_test;
mod softmax_threadgroup_test;
mod tensor_test;
