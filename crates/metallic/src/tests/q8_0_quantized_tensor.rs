#![cfg(test)]
use half::f16;

use crate::{
    Context, MetalError, tensor::{
        Q8_0_BLOCK_SIZE_BYTES, Q8_0_WEIGHTS_PER_BLOCK, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor}
    }
};

fn make_q8_0_block(scale: f16, values: &[i8; 32]) -> [u8; Q8_0_BLOCK_SIZE_BYTES] {
    let mut out = [0u8; Q8_0_BLOCK_SIZE_BYTES];
    let s = scale.to_bits().to_le_bytes();
    out[0] = s[0];
    out[1] = s[1];
    for i in 0..32 {
        out[2 + i] = values[i] as u8;
    }
    out
}

#[test]
fn builds_quantized_q8_0_tensor_without_upcast() -> Result<(), MetalError> {
    let ctx = Context::<crate::tensor::F16>::new()?;

    // 2 blocks * 32 weights = 64 logical weights
    let mut raw = Vec::with_capacity(2 * Q8_0_BLOCK_SIZE_BYTES);
    let vals0: [i8; 32] = core::array::from_fn(|i| i as i8);
    let vals1: [i8; 32] = core::array::from_fn(|i| (i as i8) - 16);
    raw.extend_from_slice(&make_q8_0_block(f16::from_f32(1.0), &vals0));
    raw.extend_from_slice(&make_q8_0_block(f16::from_f32(0.5), &vals1));

    let mut data_bytes = Vec::with_capacity(2 * Q8_0_WEIGHTS_PER_BLOCK);
    let mut scale_bytes = Vec::with_capacity(2 * Q8_0_SCALE_BYTES_PER_BLOCK);
    for chunk in raw.chunks_exact(Q8_0_BLOCK_SIZE_BYTES) {
        scale_bytes.extend_from_slice(&chunk[0..Q8_0_SCALE_BYTES_PER_BLOCK]);
        data_bytes.extend_from_slice(&chunk[Q8_0_SCALE_BYTES_PER_BLOCK..Q8_0_BLOCK_SIZE_BYTES]);
    }

    let qt = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![64], &data_bytes, &scale_bytes, &ctx)?;
    assert_eq!(qt.logical_len(), 64);
    assert_eq!(qt.blocks(), 2);
    assert_eq!(qt.raw_len_bytes(), 2 * Q8_0_WEIGHTS_PER_BLOCK);
    assert_eq!(Q8_0_WEIGHTS_PER_BLOCK, 32);
    Ok(())
}
