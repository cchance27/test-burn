use crate::metallic::Operation as _;
use crate::metallic::cache_keys::SdpaKey;
use crate::metallic::cacheable::Cacheable as _;
use crate::metallic::cacheable_sdpa::CacheableSdpa;

use crate::metallic::cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey};
use crate::metallic::kernels::softmax::SoftmaxOp;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Tensor};
use objc2_metal::MTLComputePipelineState as _;

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
mod tensor_test;
