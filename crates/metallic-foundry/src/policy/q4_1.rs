use metallic_loader::quant_spec::Q4_1_SPEC;
use metallic_macros::{GgufBlockQuantRuntime, MetalPolicy};

use crate::tensor::Dtype;

const Q4_1_DATA_BYTES: usize = 16;

#[inline]
fn q4_1_write_block(qs: &[u8], out: &mut [u8]) {
    debug_assert_eq!(qs.len(), Q4_1_DATA_BYTES);
    debug_assert_eq!(out.len(), Q4_1_DATA_BYTES);

    // GGUF/GGML Q4_1 stores qs[0..15] where low nibbles are weights 0..15 and high nibbles are weights 16..31.
    // Repack into adjacent pairs for the Metal policy's packed logical indexing.
    for i in 0..8 {
        let b0 = qs[2 * i];
        let b1 = qs[2 * i + 1];
        out[i] = ((b1 & 0x0F) << 4) | (b0 & 0x0F);
        out[i + 8] = (b1 & 0xF0) | (b0 >> 4);
    }
}

#[derive(Debug, Clone, Default, MetalPolicy, GgufBlockQuantRuntime)]
#[policy(
    header = "policies/policy_q4_1.metal",
    struct_name = "PolicyQ4_1",
    short_name = "q4_1",
    // Packed format: logical dimensions exceed byte count.
    element_size = 0,
    block_size = 32,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 20,
    weights_per_block = 32
)]
#[gguf_runtime(
    source_dtype = Dtype::Q4_1,
    scales_dtype = Dtype::Q8_0,
    spec = Q4_1_SPEC,
    scale_bytes = 4,
    data_bytes = 16,
    write_block = q4_1_write_block
)]
pub struct PolicyQ4_1;

#[cfg(test)]
mod tests {
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
}
