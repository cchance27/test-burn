#![cfg(test)]

use super::*;

#[test]
fn q4_0_canonical_reorders_and_deinterleaves_consistently() {
    const Q4_0_BLOCK_BYTES: usize = Q4_0_SPEC.block_bytes;
    const Q4_0_SCALE_BYTES: usize = 2;

    let _total_blocks = 1;
    let mut raw = vec![0u8; Q4_0_BLOCK_BYTES];
    raw[0] = 0xAA; // scale L
    raw[1] = 0xBB; // scale H
    let mut qs = [0u8; 16];
    for (i, q) in qs.iter_mut().enumerate() {
        // Low nibble represents weights 0..15; high nibble represents weights 16..31.
        // Make them distinct so the transform is observable.
        *q = (i as u8 & 0x0F) | (((i as u8 ^ 0x0F) & 0x0F) << 4);
    }
    raw[2..].copy_from_slice(&qs);

    let mut data_out = vec![0u8; 16];
    let mut scales_out = vec![0u8; 2];

    crate::policy::block_quant::split_blocks::<Q4_0_BLOCK_BYTES, Q4_0_SCALE_BYTES, Q4_0_DATA_BYTES>(
        &raw,
        1,
        1,
        false,
        &mut data_out,
        &mut scales_out,
        q4_0_write_block,
    );

    assert_eq!(scales_out[0], 0xAA);
    assert_eq!(scales_out[1], 0xBB);

    // After transform:
    // - data_out[0..8] packs low nibbles of qs[0..16] into adjacent pairs
    // - data_out[8..16] packs high nibbles similarly
    let mut expected = vec![0u8; 16];
    q4_0_write_block(&qs, &mut expected);
    assert_eq!(data_out, expected);
}
