#![cfg(test)]

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
