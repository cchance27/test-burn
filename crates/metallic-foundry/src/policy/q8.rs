use metallic_loader::quant_spec::Q8_0_SPEC;
use metallic_macros::{GgufBlockQuantRuntime, MetalPolicy};

use crate::tensor::Dtype;

const Q8_0_DATA_BYTES: usize = 32;

#[inline]
fn q8_0_write_block(qs: &[u8], out: &mut [u8]) {
    debug_assert_eq!(qs.len(), Q8_0_DATA_BYTES);
    debug_assert_eq!(out.len(), Q8_0_DATA_BYTES);
    out.copy_from_slice(qs);
}

#[derive(Debug, Clone, Default, MetalPolicy, GgufBlockQuantRuntime)]
#[policy(
    header = "policies/policy_q8.metal",
    struct_name = "PolicyQ8",
    short_name = "q8",
    element_size = 1,
    block_size = 32,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 34,
    weights_per_block = 32
)]
#[gguf_runtime(
    source_dtype = Dtype::Q8_0,
    scales_dtype = Dtype::Q8_0,
    spec = Q8_0_SPEC,
    scale_bytes = 2,
    data_bytes = 32,
    write_block = q8_0_write_block
)]
pub struct PolicyQ8;

#[path = "q8.test.rs"]
mod tests;
