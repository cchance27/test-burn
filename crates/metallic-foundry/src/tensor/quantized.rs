/// Swizzle NK row-major Q8_0 bytes (rows = output dim, cols = input dim) into a layout where
/// all rows for a given K-block are contiguous, improving streaming efficiency.
/// Returns `None` when the provided dimensions do not match the byte length.
// DEBT: Only used in tests? might not be needed anymore.
pub fn swizzle_q8_0_blocks_nk(rows_n: usize, cols_k: usize, raw_bytes: &[u8]) -> Option<Vec<u8>> {
    let blocks_per_row = cols_k.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
    let expected_blocks = rows_n.checked_mul(blocks_per_row)?;
    let expected_bytes = expected_blocks.checked_mul(Q8_0_BLOCK_SIZE_BYTES)?;
    if expected_bytes != raw_bytes.len() {
        return None;
    }

    let mut swizzled = vec![0u8; raw_bytes.len()];
    for k_block in 0..blocks_per_row {
        for row in 0..rows_n {
            let src_block = row * blocks_per_row + k_block;
            let dst_block = k_block * rows_n + row;
            let src = src_block * Q8_0_BLOCK_SIZE_BYTES;
            let dst = dst_block * Q8_0_BLOCK_SIZE_BYTES;
            swizzled[dst..dst + Q8_0_BLOCK_SIZE_BYTES].copy_from_slice(&raw_bytes[src..src + Q8_0_BLOCK_SIZE_BYTES]);
        }
    }

    Some(swizzled)
}

/// Q8_0 block (zero-copy layout): 2 bytes f16 scale + 32 bytes i8 values.
/// This is primarily documentation for the packed format; we store raw bytes in `Tensor<U8>`.
// DEBT: this isn't used anywhere i dont think anymore.
#[repr(C, packed)]
pub struct Q8_0Block {
    pub scale_f16_le: [u8; 2],
    pub qs: [i8; 32],
}

// DEBT: these should likely just be moved to the policy Q8 to centralize it.
pub const Q8_0_BLOCK_SIZE_BYTES: usize = 34;
pub const Q8_0_WEIGHTS_PER_BLOCK: usize = 32;
pub const Q8_0_SCALE_BYTES_PER_BLOCK: usize = 2;
