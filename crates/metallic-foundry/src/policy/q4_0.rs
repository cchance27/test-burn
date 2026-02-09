use metallic_loader::quant_spec::Q4_0_SPEC;
use metallic_macros::{GgufBlockQuantRuntime, MetalPolicy};

use crate::tensor::Dtype;

const Q4_0_DATA_BYTES: usize = 16;

#[inline]
fn q4_0_write_block(qs: &[u8], out: &mut [u8]) {
    debug_assert_eq!(qs.len(), Q4_0_DATA_BYTES);
    debug_assert_eq!(out.len(), Q4_0_DATA_BYTES);

    // GGUF/GGML Q4_0 stores qs[0..15] where low nibbles are weights 0..15 and high nibbles are weights 16..31.
    // The Metal policy expects adjacent-pair packing: out[j] = (w[2j+1] << 4) | w[2j].
    for i in 0..8 {
        let b0 = qs[2 * i];
        let b1 = qs[2 * i + 1];
        out[i] = ((b1 & 0x0F) << 4) | (b0 & 0x0F);
        out[i + 8] = (b1 & 0xF0) | (b0 >> 4);
    }
}

#[derive(Debug, Clone, Default, MetalPolicy, GgufBlockQuantRuntime)]
#[policy(
    header = "policies/policy_q4_0.metal",
    struct_name = "PolicyQ4_0",
    short_name = "q4_0",
    // We set element_size to 0 for packed types to bypass strict byte-count validation
    // in the executor's layout check, as logical dimensions > physical bytes.
    element_size = 0,
    block_size = 32,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 18,
    weights_per_block = 32
)]
#[gguf_runtime(
    source_dtype = Dtype::Q4_0,
    scales_dtype = Dtype::Q8_0,
    spec = Q4_0_SPEC,
    scale_bytes = 2,
    data_bytes = 16,
    write_block = q4_0_write_block
)]
pub struct PolicyQ4_0;

#[cfg(test)]
mod tests {
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
}
