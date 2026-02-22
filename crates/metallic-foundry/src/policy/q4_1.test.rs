#![cfg(test)]

use super::*;

#[test]
fn q4_1_canonical_reorders_scales_and_deinterleaves_consistently() {
    const Q4_1_BLOCK_BYTES: usize = Q4_1_SPEC.block_bytes;
    const Q4_1_SCALE_BYTES: usize = 4;

    let mut raw = vec![0u8; Q4_1_BLOCK_BYTES];
    raw[0] = 0x11; // d low
    raw[1] = 0x22; // d high
    raw[2] = 0x33; // m low
    raw[3] = 0x44; // m high

    let mut qs = [0u8; 16];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = (i as u8 & 0x0F) | (((i as u8 ^ 0x0F) & 0x0F) << 4);
    }
    raw[Q4_1_SCALE_BYTES..].copy_from_slice(&qs);

    let mut data_out = vec![0u8; 16];
    let mut scales_out = vec![0u8; 4];

    crate::policy::block_quant::split_blocks::<Q4_1_BLOCK_BYTES, Q4_1_SCALE_BYTES, Q4_1_DATA_BYTES>(
        &raw,
        1,
        1,
        false,
        &mut data_out,
        &mut scales_out,
        q4_1_write_block,
    );

    assert_eq!(&scales_out, &[0x11, 0x22, 0x33, 0x44]);

    let mut expected = vec![0u8; 16];
    q4_1_write_block(&qs, &mut expected);
    assert_eq!(data_out, expected);
}
