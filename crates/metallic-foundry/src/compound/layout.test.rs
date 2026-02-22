#![cfg(test)]

use super::*;

#[test]
fn test_sdpa_scratch_padding() {
    // Tile size 32
    let layout = TiledLayout::sdpa_scratch(8, 10, 128, 32);
    // M=10 -> padded_m=32
    assert_eq!(layout.padded_m, 32);
    assert_eq!(layout.row_stride, 128);
    assert_eq!(layout.head_stride, 32 * 128);

    // M=1 -> padded_m=1 (no padding for decode)
    let decode = TiledLayout::sdpa_scratch(8, 1, 128, 32);
    assert_eq!(decode.padded_m, 1);
    assert_eq!(decode.head_stride, 128);
}
