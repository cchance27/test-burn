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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_canonical_reorders_scales_and_data_consistently() {
        const Q8_0_WPB: usize = Q8_0_SPEC.weights_per_block;
        const Q8_0_BLOCK_BYTES: usize = Q8_0_SPEC.block_bytes;
        const Q8_0_SCALE_BYTES: usize = 2;

        let target_n = 3;
        let blocks_per_k = 2;
        let total_blocks = target_n * blocks_per_k;

        let mut raw = vec![0u8; total_blocks * Q8_0_BLOCK_BYTES];
        for src_block_idx in 0..total_blocks {
            let start = src_block_idx * Q8_0_BLOCK_BYTES;
            raw[start] = src_block_idx as u8; // scale byte 0
            raw[start + 1] = 0xEE; // scale byte 1 sentinel
            raw[start + Q8_0_SCALE_BYTES..start + Q8_0_BLOCK_BYTES].fill((0x10 + src_block_idx) as u8);
        }

        let mut data_out = vec![0u8; total_blocks * Q8_0_WPB];
        let mut scales_out = vec![0u8; total_blocks * Q8_0_SCALE_BYTES];

        crate::policy::block_quant::split_blocks::<Q8_0_BLOCK_BYTES, Q8_0_SCALE_BYTES, Q8_0_DATA_BYTES>(
            &raw,
            blocks_per_k,
            target_n,
            true,
            &mut data_out,
            &mut scales_out,
            q8_0_write_block,
        );

        for src_block_idx in 0..total_blocks {
            let scale_start = src_block_idx * Q8_0_SCALE_BYTES;
            assert_eq!(scales_out[scale_start], src_block_idx as u8);
            assert_eq!(scales_out[scale_start + 1], 0xEE);

            let dst = crate::policy::block_quant::canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n);
            let data_start = dst * Q8_0_WPB;
            assert_eq!(data_out[data_start], (0x10 + src_block_idx) as u8);
            assert_eq!(data_out[data_start + Q8_0_WPB - 1], (0x10 + src_block_idx) as u8);
        }
    }
}
