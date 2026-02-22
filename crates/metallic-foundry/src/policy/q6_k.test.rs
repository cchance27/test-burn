#![cfg(test)]

use super::*;

#[test]
fn q6_k_decode_block_unpacks_expected_q_and_scale() {
    let mut raw = [0u8; Q6_K_SOURCE_BLOCK_BYTES];
    // scales = [1, 2, 3, ... 16]
    for i in 0..16 {
        raw[192 + i] = (i as i8 + 1) as u8;
    }
    // d = 1.0f16 (tail bytes)
    raw[Q6_K_SOURCE_BLOCK_BYTES - 2] = 0x00;
    raw[Q6_K_SOURCE_BLOCK_BYTES - 1] = 0x3C;

    // ql/qh = all zeros -> q should decode to -32
    let mut q_vals = [0u8; Q6_K_SOURCE_WPB];
    let mut scales = [0u16; 16];
    decode_q6_k_block(&raw, &mut q_vals, &mut scales);

    assert!(q_vals.iter().all(|&v| v == (-(32_i8) as u8)));
    assert_eq!(f16::from_bits(scales[0]).to_f32(), 1.0);
    assert_eq!(f16::from_bits(scales[15]).to_f32(), 16.0);
}
