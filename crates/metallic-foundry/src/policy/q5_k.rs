use half::f16;
use metallic_loader::{LoadedModel, quant_spec::Q5_K_SPEC};
use metallic_macros::MetalPolicy;

use super::{LoaderStage, MetalPolicyRuntime};
use crate::{
    Foundry, compound::Layout, policy::block_quant::canonical_dst_block_idx, spec::{FastBindings, ResolvedSymbols}, tensor::Dtype, types::{MetalResourceOptions, TensorArg}
};

const Q5_K_SOURCE_WPB: usize = Q5_K_SPEC.weights_per_block;
const Q5_K_SOURCE_BLOCK_BYTES: usize = Q5_K_SPEC.block_bytes; // d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
const Q5_K_DECODED_WPB: usize = 16;
const Q5_K_DECODED_DATA_BYTES: usize = 16; // u8[16], each value in [0,31]
const Q5_K_DECODED_SCALE_BYTES: usize = 8; // f32(scale) + f32(affine) per 16 weights

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8], d: &mut u8, m: &mut u8) {
    debug_assert_eq!(scales.len(), 12);
    if j < 4 {
        *d = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

#[inline]
fn decode_q5_k_block(
    raw_block: &[u8],
    q_out: &mut [u8; Q5_K_SOURCE_WPB],
    scales_out: &mut [f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB],
    affine_out: &mut [f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB],
) {
    debug_assert_eq!(raw_block.len(), Q5_K_SOURCE_BLOCK_BYTES);

    let d = f16::from_bits(u16::from(raw_block[0]) | (u16::from(raw_block[1]) << 8)).to_f32();
    let dmin = f16::from_bits(u16::from(raw_block[2]) | (u16::from(raw_block[3]) << 8)).to_f32();
    let scales = &raw_block[4..16];
    let qh = &raw_block[16..48];
    let ql = &raw_block[48..176];

    // GGML Q5_K stores scale/min per 32 weights; Foundry decoded layout uses 16 weights/block.
    // Duplicate each 32-group's scale/min for its two 16-weight child groups.
    for group32 in 0..8 {
        let mut sc = 0u8;
        let mut mn = 0u8;
        get_scale_min_k4(group32, scales, &mut sc, &mut mn);
        let scale_val = d * sc as f32;
        let affine_val = -(dmin * mn as f32);
        let g16 = group32 * 2;
        scales_out[g16] = scale_val;
        scales_out[g16 + 1] = scale_val;
        affine_out[g16] = affine_val;
        affine_out[g16 + 1] = affine_val;
    }

    // Mirrors ggml::dequantize_row_q5_K indexing, keeping q-values unscaled in [0,31].
    for n in 0..4 {
        let ql_n = &ql[n * 32..n * 32 + 32];
        let u1 = 1u8 << (2 * n);
        let u2 = 2u8 << (2 * n);
        let base = n * 64;

        for l in 0..32 {
            let lo = ql_n[l] & 0x0F;
            let hi = ql_n[l] >> 4;
            q_out[base + l] = lo + if (qh[l] & u1) != 0 { 16 } else { 0 };
            q_out[base + l + 32] = hi + if (qh[l] & u2) != 0 { 16 } else { 0 };
        }
    }
}

#[derive(Debug, Clone, Default, MetalPolicy)]
#[policy(
    header = "policies/policy_q5_k.metal",
    struct_name = "PolicyQ5K",
    short_name = "q5_k",
    element_size = 1,
    block_size = 16,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 24,
    weights_per_block = 16
)]
pub struct PolicyQ5K;

impl LoaderStage for PolicyQ5K {
    fn params_struct(&self) -> String {
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        let weight = fast_bindings.get(resolved.weights).expect("Q5_K weight bound");
        let scales_idx = resolved.scales.expect("Q5_K requires scales index");
        let scales = fast_bindings.get(scales_idx).expect("Q5_K scales bound");
        smallvec![weight.clone(), scales.clone()]
    }

    fn quantization_type(&self) -> std::sync::Arc<dyn MetalPolicyRuntime> {
        std::sync::Arc::new(PolicyQ5K)
    }
}

impl MetalPolicyRuntime for PolicyQ5K {
    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(self.clone())
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        model: &dyn LoadedModel,
        source_tensor_name: &str,
        logical_name: &str,
        layout: Layout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let tensor_info = model
            .tensor_info(source_tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in model", source_tensor_name))?;

        let dims: Vec<usize> = tensor_info.dimensions.clone();
        if dims.len() != 2 {
            return Err(anyhow::anyhow!("Q5_K tensor '{}' must be 2D (got {:?})", source_tensor_name, dims));
        }

        let (target_k, target_n, is_canonical) = match layout {
            Layout::RowMajor => (dims[1], dims[0], false),
            Layout::ColMajor => (dims[0], dims[1], false),
            Layout::Canonical { expected_k, expected_n } => (expected_k, expected_n, true),
        };

        let tensor_data_guard = model.tensor_data(source_tensor_name)?;
        let raw = tensor_data_guard.as_slice();

        if tensor_info.data_type != Dtype::Q5_K {
            return Err(anyhow::anyhow!(
                "Tensor '{}' dtype mismatch: expected {:?}, got {:?}",
                source_tensor_name,
                Dtype::Q5_K,
                tensor_info.data_type
            ));
        }

        let (source_k_len, source_n_len) = if is_canonical {
            if !((dims[0] == target_k && dims[1] == target_n) || (dims[0] == target_n && dims[1] == target_k)) {
                return Err(anyhow::anyhow!(
                    "Canonical Q5_K tensor '{}' dims {:?} mismatch (K,N)=({},{})",
                    source_tensor_name,
                    dims,
                    target_k,
                    target_n
                ));
            }
            (target_k, target_n)
        } else {
            // GGUF quant blocks are packed along ne[0] (K axis), with rows over ne[1].
            // Embedding tables are looked up as [vocab, d_model] in embedding.metal (tok-major rows),
            // so keep token_embd row-major orientation aligned with runtime lookup.
            let is_embedding_table = logical_name == "embedding" || source_tensor_name == "token_embd.weight";
            match layout {
                Layout::RowMajor if is_embedding_table => (dims[1], dims[0]),
                Layout::RowMajor => (dims[0], dims[1]),
                Layout::ColMajor => (dims[1], dims[0]),
                Layout::Canonical { .. } => unreachable!("handled above"),
            }
        };

        if raw.len() % Q5_K_SOURCE_BLOCK_BYTES != 0 {
            return Err(anyhow::anyhow!(
                "Q5_K tensor '{}' byte size {} is not a multiple of block bytes {}",
                source_tensor_name,
                raw.len(),
                Q5_K_SOURCE_BLOCK_BYTES
            ));
        }

        if source_n_len == 0 {
            return Err(anyhow::anyhow!("Q5_K tensor '{}' has zero rows", source_tensor_name));
        }

        let total_src_blocks = raw.len() / Q5_K_SOURCE_BLOCK_BYTES;
        if !total_src_blocks.is_multiple_of(source_n_len) {
            return Err(anyhow::anyhow!(
                "Q5_K tensor '{}' source blocks {} not divisible by rows {}",
                source_tensor_name,
                total_src_blocks,
                source_n_len
            ));
        }

        let src_blocks_per_k = total_src_blocks / source_n_len;
        let expected_src_blocks_per_k = source_k_len.div_ceil(Q5_K_SOURCE_WPB);
        if src_blocks_per_k != expected_src_blocks_per_k {
            return Err(anyhow::anyhow!(
                "Q5_K tensor '{}' blocks-per-row mismatch: got {}, expected {} from K={}",
                source_tensor_name,
                src_blocks_per_k,
                expected_src_blocks_per_k,
                source_k_len
            ));
        }

        let dst_blocks_per_k = source_k_len.div_ceil(Q5_K_DECODED_WPB);
        if src_blocks_per_k * (Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB) < dst_blocks_per_k {
            return Err(anyhow::anyhow!(
                "Q5_K tensor '{}' has insufficient source blocks: src_blocks_per_k={}, dst_blocks_per_k={}",
                source_tensor_name,
                src_blocks_per_k,
                dst_blocks_per_k
            ));
        }
        let n_dim_len = source_n_len;
        let total_dst_blocks = n_dim_len * dst_blocks_per_k;
        let data_len = total_dst_blocks * Q5_K_DECODED_DATA_BYTES;
        let scales_len = total_dst_blocks * Q5_K_DECODED_SCALE_BYTES;

        let data_buffer = foundry
            .device
            .new_buffer(data_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q5_K data"))?;
        let scales_buffer = foundry
            .device
            .new_buffer(scales_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q5_K scales"))?;

        let layout_hint = model
            .metadata()
            .get_string("metallic.gguf.layout_hint")
            .unwrap_or(std::borrow::Cow::Borrowed("nk"));
        if is_canonical && layout_hint != "nk" {
            return Err(anyhow::anyhow!("Canonical Q5_K load requires Nk layout"));
        }

        data_buffer.write_via_slice(data_len, |data_out| {
            scales_buffer.write_via_slice(scales_len, |scales_out| {
                let mut q_vals = [0u8; Q5_K_SOURCE_WPB];
                let mut scale_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
                let mut affine_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];

                for row in 0..n_dim_len {
                    for src_block in 0..src_blocks_per_k {
                        let src_block_idx = row * src_blocks_per_k + src_block;
                        let src_off = src_block_idx * Q5_K_SOURCE_BLOCK_BYTES;
                        let chunk = &raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES];
                        decode_q5_k_block(chunk, &mut q_vals, &mut scale_vals, &mut affine_vals);

                        let valid_values = (source_k_len.saturating_sub(src_block * Q5_K_SOURCE_WPB)).min(Q5_K_SOURCE_WPB);
                        let valid_groups = valid_values.div_ceil(Q5_K_DECODED_WPB);
                        for g in 0..valid_groups {
                            let src_block16_idx = row * dst_blocks_per_k + src_block * 16 + g;
                            let dst_weight_block_idx = if is_canonical {
                                canonical_dst_block_idx(src_block16_idx, dst_blocks_per_k, target_n)
                            } else {
                                src_block16_idx
                            };

                            let q_src = g * Q5_K_DECODED_DATA_BYTES;
                            let q_dst = dst_weight_block_idx * Q5_K_DECODED_DATA_BYTES;
                            data_out[q_dst..q_dst + Q5_K_DECODED_DATA_BYTES]
                                .copy_from_slice(&q_vals[q_src..q_src + Q5_K_DECODED_DATA_BYTES]);

                            // Keep scales/affine in row-major block order (like Q8/Q6 loaders).
                            // WEIGHT_INDEX handles canonical swizzle for weight bytes, but scale indexing
                            // in kernels uses (row * blocks_per_k + block), i.e. unswizzled block order.
                            let s_dst = src_block16_idx * Q5_K_DECODED_SCALE_BYTES;
                            let s = scale_vals[g].to_le_bytes();
                            let a = affine_vals[g].to_le_bytes();
                            scales_out[s_dst..s_dst + 4].copy_from_slice(&s);
                            scales_out[s_dst + 4..s_dst + 8].copy_from_slice(&a);
                        }
                    }
                }
            });
        });

        let weights = TensorArg::from_buffer(
            data_buffer,
            Dtype::Q5_K,
            if is_canonical {
                vec![total_dst_blocks * Q5_K_DECODED_WPB]
            } else {
                dims.clone()
            },
            if is_canonical {
                vec![1]
            } else {
                crate::tensor::compute_strides(&dims)
            },
        );
        let scales = TensorArg::from_buffer(scales_buffer, Dtype::Q8_0, vec![scales_len], vec![1]);

        Ok(vec![
            (logical_name.to_string(), weights),
            (format!("{}_scales", logical_name), scales),
        ])
    }
}

#[cfg(test)]
mod tests {
    use metallic_loader::ModelLoader;

    use super::*;
    use crate::{compound::Layout, policy::block_quant::canonical_dst_block_idx};

    fn dequant_direct_ggml_style(raw: &[u8; Q5_K_SOURCE_BLOCK_BYTES]) -> [f32; Q5_K_SOURCE_WPB] {
        let mut out = [0.0f32; Q5_K_SOURCE_WPB];
        let d = f16::from_bits(u16::from(raw[0]) | (u16::from(raw[1]) << 8)).to_f32();
        let dmin = f16::from_bits(u16::from(raw[2]) | (u16::from(raw[3]) << 8)).to_f32();
        let scales = &raw[4..16];
        let qh = &raw[16..48];
        let ql = &raw[48..176];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..Q5_K_SOURCE_WPB).step_by(64) {
            let mut sc = 0u8;
            let mut mn = 0u8;
            get_scale_min_k4(is, scales, &mut sc, &mut mn);
            let d1 = d * sc as f32;
            let m1 = dmin * mn as f32;

            get_scale_min_k4(is + 1, scales, &mut sc, &mut mn);
            let d2 = d * sc as f32;
            let m2 = dmin * mn as f32;

            let ql32 = &ql[(j / 2)..(j / 2 + 32)];
            for l in 0..32 {
                let q = (ql32[l] & 0x0F) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[j + l] = d1 * q as f32 - m1;
            }
            for l in 0..32 {
                let q = (ql32[l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[j + 32 + l] = d2 * q as f32 - m2;
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        out
    }

    fn decode_q5_k_block_reference(
        raw_block: &[u8; Q5_K_SOURCE_BLOCK_BYTES],
        q_out: &mut [u8; Q5_K_SOURCE_WPB],
        scales_out: &mut [f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB],
        affine_out: &mut [f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB],
    ) {
        let d = f16::from_bits(u16::from(raw_block[0]) | (u16::from(raw_block[1]) << 8)).to_f32();
        let dmin = f16::from_bits(u16::from(raw_block[2]) | (u16::from(raw_block[3]) << 8)).to_f32();
        let s = &raw_block[4..16];
        let qh = &raw_block[16..48];
        let ql = &raw_block[48..176];

        // Extract 8x (scale,min), one pair per 32 values.
        // This is intentionally written as a direct reference path (no helper calls)
        // so the test can catch packing bugs in get_scale_min_k4/decode_q5_k_block.
        for j in 0..8 {
            let (sc, mn) = if j < 4 {
                (s[j] & 0x3F, s[j + 4] & 0x3F)
            } else {
                let sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
                let mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
                (sc, mn)
            };

            let scale_val = d * sc as f32;
            let affine_val = -(dmin * mn as f32);
            let g16 = j * 2;
            scales_out[g16] = scale_val;
            scales_out[g16 + 1] = scale_val;
            affine_out[g16] = affine_val;
            affine_out[g16 + 1] = affine_val;
        }

        // GGML dequant loop shape: 4 chunks of 64 values, split from 32 ql bytes and 32 qh bytes.
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let base = chunk * 64;
            let ql_chunk = &ql[chunk * 32..chunk * 32 + 32];
            for l in 0..32 {
                let lo = ql_chunk[l] & 0x0F;
                let hi = ql_chunk[l] >> 4;
                q_out[base + l] = lo + if (qh[l] & u1) != 0 { 16 } else { 0 };
                q_out[base + l + 32] = hi + if (qh[l] & u2) != 0 { 16 } else { 0 };
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    #[test]
    fn q5_k_decode_block_unpacks_expected_q_scale_and_affine() {
        let mut raw = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
        // d = 2.0, dmin = 1.0
        raw[0] = 0x00;
        raw[1] = 0x40;
        raw[2] = 0x00;
        raw[3] = 0x3C;

        // Encode j=0 scale/min directly in low 6 bits.
        raw[4] = 3; // scale group 0
        raw[8] = 5; // min group 0

        // ql/qh all zeros -> q values decode to 0.
        let mut q_vals = [0u8; Q5_K_SOURCE_WPB];
        let mut scales = [0.0f32; 16];
        let mut affines = [0.0f32; 16];
        decode_q5_k_block(&raw, &mut q_vals, &mut scales, &mut affines);

        assert!(q_vals.iter().all(|&v| v == 0));
        // Group 0 duplicates across first two 16-weight groups.
        assert_eq!(scales[0], 6.0);
        assert_eq!(scales[1], 6.0);
        assert_eq!(affines[0], -5.0);
        assert_eq!(affines[1], -5.0);
    }

    #[test]
    fn q5_k_decode_matches_direct_ggml_elementwise() {
        // Deterministic xorshift for reproducible fuzzing without extra deps.
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let mut seed = 0x1234_5678u32;
        for _case in 0..128 {
            let mut raw = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
            for b in &mut raw {
                *b = (next_u32(&mut seed) & 0xFF) as u8;
            }
            // Keep d/dmin finite and moderate so comparisons are stable and meaningful.
            let d = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (4.0 / 255.0);
            let dmin = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (4.0 / 255.0);
            let d_bits = f16::from_f32(d).to_bits();
            let dm_bits = f16::from_f32(dmin).to_bits();
            raw[0] = (d_bits & 0xFF) as u8;
            raw[1] = (d_bits >> 8) as u8;
            raw[2] = (dm_bits & 0xFF) as u8;
            raw[3] = (dm_bits >> 8) as u8;

            let mut q_vals = [0u8; Q5_K_SOURCE_WPB];
            let mut scales = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            let mut affines = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            decode_q5_k_block(&raw, &mut q_vals, &mut scales, &mut affines);

            let direct = dequant_direct_ggml_style(&raw);
            for i in 0..Q5_K_SOURCE_WPB {
                let g = i / Q5_K_DECODED_WPB;
                let scale = scales[g];
                let affine = affines[g];
                let ours = scale * q_vals[i] as f32 + affine;
                let diff = (ours - direct[i]).abs();
                assert!(
                    diff <= 1.0e-5,
                    "mismatch at idx={i}: ours={ours:.6} direct={:.6} diff={diff:.6}",
                    direct[i]
                );
            }
        }
    }

    #[test]
    fn q5_k_decode_matches_independent_reference_unpack() {
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let mut seed = 0xA11C_E5EDu32;
        for _ in 0..256 {
            let mut raw = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
            for b in &mut raw {
                *b = (next_u32(&mut seed) & 0xFF) as u8;
            }

            // Keep d and dmin finite and representative.
            let d = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (8.0 / 255.0);
            let dmin = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (8.0 / 255.0);
            let d_bits = f16::from_f32(d).to_bits();
            let dm_bits = f16::from_f32(dmin).to_bits();
            raw[0] = (d_bits & 0xFF) as u8;
            raw[1] = (d_bits >> 8) as u8;
            raw[2] = (dm_bits & 0xFF) as u8;
            raw[3] = (dm_bits >> 8) as u8;

            let mut q_a = [0u8; Q5_K_SOURCE_WPB];
            let mut s_a = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            let mut a_a = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            decode_q5_k_block(&raw, &mut q_a, &mut s_a, &mut a_a);

            let mut q_b = [0u8; Q5_K_SOURCE_WPB];
            let mut s_b = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            let mut a_b = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
            decode_q5_k_block_reference(&raw, &mut q_b, &mut s_b, &mut a_b);

            assert_eq!(q_a, q_b, "q unpack mismatch");
            for i in 0..s_a.len() {
                assert!((s_a[i] - s_b[i]).abs() <= 1e-6, "scale mismatch at {i}");
                assert!((a_a[i] - a_b[i]).abs() <= 1e-6, "affine mismatch at {i}");
            }
        }
    }

    #[test]
    fn q5_k_canonical_repack_matches_direct_ggml_matmul() {
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let mut seed = 0x51A5_9EEDu32;
        let k = 256usize * 2 + 48; // exercise partial trailing block
        let n = 7usize;
        let src_blocks_per_k = k.div_ceil(Q5_K_SOURCE_WPB);
        let dst_blocks_per_k = k.div_ceil(Q5_K_DECODED_WPB);
        let raw_len = n * src_blocks_per_k * Q5_K_SOURCE_BLOCK_BYTES;
        let mut raw = vec![0u8; raw_len];

        // Build finite/randomized raw q5 blocks.
        for row in 0..n {
            for sb in 0..src_blocks_per_k {
                let off = (row * src_blocks_per_k + sb) * Q5_K_SOURCE_BLOCK_BYTES;
                let blk = &mut raw[off..off + Q5_K_SOURCE_BLOCK_BYTES];
                for b in blk.iter_mut() {
                    *b = (next_u32(&mut seed) & 0xFF) as u8;
                }

                // Keep d/dmin finite and moderate.
                let d = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (6.0 / 255.0);
                let dmin = 0.01f32 + ((next_u32(&mut seed) & 0xFF) as f32) * (6.0 / 255.0);
                let d_bits = f16::from_f32(d).to_bits();
                let dm_bits = f16::from_f32(dmin).to_bits();
                blk[0] = (d_bits & 0xFF) as u8;
                blk[1] = (d_bits >> 8) as u8;
                blk[2] = (dm_bits & 0xFF) as u8;
                blk[3] = (dm_bits >> 8) as u8;
            }
        }

        // Repack exactly like load_weights canonical path does.
        let total_dst_blocks = n * dst_blocks_per_k;
        let mut data_out = vec![0u8; total_dst_blocks * Q5_K_DECODED_DATA_BYTES];
        let mut scales_out = vec![0u8; total_dst_blocks * Q5_K_DECODED_SCALE_BYTES];

        let mut q_vals = [0u8; Q5_K_SOURCE_WPB];
        let mut scale_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
        let mut affine_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];

        for row in 0..n {
            for src_block in 0..src_blocks_per_k {
                let src_block_idx = row * src_blocks_per_k + src_block;
                let src_off = src_block_idx * Q5_K_SOURCE_BLOCK_BYTES;
                let chunk = &raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES];
                decode_q5_k_block(chunk, &mut q_vals, &mut scale_vals, &mut affine_vals);

                let valid_values = (k.saturating_sub(src_block * Q5_K_SOURCE_WPB)).min(Q5_K_SOURCE_WPB);
                let valid_groups = valid_values.div_ceil(Q5_K_DECODED_WPB);
                for g in 0..valid_groups {
                    let src_block16_idx = row * dst_blocks_per_k + src_block * 16 + g;
                    let dst_weight_block_idx = canonical_dst_block_idx(src_block16_idx, dst_blocks_per_k, n);

                    let q_src = g * Q5_K_DECODED_DATA_BYTES;
                    let q_dst = dst_weight_block_idx * Q5_K_DECODED_DATA_BYTES;
                    data_out[q_dst..q_dst + Q5_K_DECODED_DATA_BYTES].copy_from_slice(&q_vals[q_src..q_src + Q5_K_DECODED_DATA_BYTES]);

                    // Row-major scale/affine block order.
                    let s_dst = src_block16_idx * Q5_K_DECODED_SCALE_BYTES;
                    scales_out[s_dst..s_dst + 4].copy_from_slice(&scale_vals[g].to_le_bytes());
                    scales_out[s_dst + 4..s_dst + 8].copy_from_slice(&affine_vals[g].to_le_bytes());
                }
            }
        }

        // Random input vector.
        let mut x = vec![0.0f32; k];
        for xv in &mut x {
            *xv = ((next_u32(&mut seed) as i32 % 2000) as f32) / 1000.0 - 1.0;
        }

        // Reference: direct GGML dequant + dot.
        let mut y_ref = vec![0.0f32; n];
        for (row, y) in y_ref.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for src_block in 0..src_blocks_per_k {
                let src_off = (row * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                let deq = dequant_direct_ggml_style(&blk);
                let k0 = src_block * Q5_K_SOURCE_WPB;
                let valid = (k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                for i in 0..valid {
                    acc += deq[i] * x[k0 + i];
                }
            }
            *y = acc;
        }

        // Canonical packed path (policy-side reconstruction).
        let mut y_can = vec![0.0f32; n];
        for (row, y) in y_can.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (kk, xv) in x.iter().enumerate().take(k) {
                let block = kk / Q5_K_DECODED_WPB;
                let elem = kk % Q5_K_DECODED_WPB;

                let dst_weight_block_idx = block * n + row; // canonical [block][row]
                let q = data_out[dst_weight_block_idx * Q5_K_DECODED_DATA_BYTES + elem] as f32;

                let s_idx = (row * dst_blocks_per_k + block) * Q5_K_DECODED_SCALE_BYTES;
                let s = f32::from_le_bytes([
                    scales_out[s_idx],
                    scales_out[s_idx + 1],
                    scales_out[s_idx + 2],
                    scales_out[s_idx + 3],
                ]);
                let a = f32::from_le_bytes([
                    scales_out[s_idx + 4],
                    scales_out[s_idx + 5],
                    scales_out[s_idx + 6],
                    scales_out[s_idx + 7],
                ]);
                acc += (s * q + a) * *xv;
            }
            *y = acc;
        }

        for i in 0..n {
            let diff = (y_ref[i] - y_can[i]).abs();
            assert!(diff <= 5.0e-4, "row {i} mismatch: ref={} can={} diff={}", y_ref[i], y_can[i], diff);
        }
    }

    fn repack_rowmajor_q5(raw: &[u8], source_k: usize, source_n: usize) -> (Vec<u8>, Vec<u8>, usize) {
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let dst_blocks_per_k = source_k.div_ceil(Q5_K_DECODED_WPB);
        let total_dst_blocks = source_n * dst_blocks_per_k;

        let mut data_out = vec![0u8; total_dst_blocks * Q5_K_DECODED_DATA_BYTES];
        let mut scales_out = vec![0u8; total_dst_blocks * Q5_K_DECODED_SCALE_BYTES];

        let mut q_vals = [0u8; Q5_K_SOURCE_WPB];
        let mut scale_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];
        let mut affine_vals = [0.0f32; Q5_K_SOURCE_WPB / Q5_K_DECODED_WPB];

        for row in 0..source_n {
            for src_block in 0..src_blocks_per_k {
                let src_block_idx = row * src_blocks_per_k + src_block;
                let src_off = src_block_idx * Q5_K_SOURCE_BLOCK_BYTES;
                let chunk = &raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES];
                decode_q5_k_block(chunk, &mut q_vals, &mut scale_vals, &mut affine_vals);

                let valid_values = (source_k.saturating_sub(src_block * Q5_K_SOURCE_WPB)).min(Q5_K_SOURCE_WPB);
                let valid_groups = valid_values.div_ceil(Q5_K_DECODED_WPB);
                for g in 0..valid_groups {
                    let block16_idx = row * dst_blocks_per_k + src_block * 16 + g;
                    let q_src = g * Q5_K_DECODED_DATA_BYTES;
                    let q_dst = block16_idx * Q5_K_DECODED_DATA_BYTES;
                    data_out[q_dst..q_dst + Q5_K_DECODED_DATA_BYTES].copy_from_slice(&q_vals[q_src..q_src + Q5_K_DECODED_DATA_BYTES]);

                    let s_dst = block16_idx * Q5_K_DECODED_SCALE_BYTES;
                    scales_out[s_dst..s_dst + 4].copy_from_slice(&scale_vals[g].to_le_bytes());
                    scales_out[s_dst + 4..s_dst + 8].copy_from_slice(&affine_vals[g].to_le_bytes());
                }
            }
        }

        (data_out, scales_out, dst_blocks_per_k)
    }

    fn direct_ggml_row_dot(raw: &[u8], source_k: usize, source_n: usize, x: &[f32]) -> Vec<f32> {
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let mut y = vec![0.0f32; source_n];
        for row in 0..source_n {
            let mut acc = 0.0f32;
            for src_block in 0..src_blocks_per_k {
                let src_off = (row * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                let deq = dequant_direct_ggml_style(&blk);
                let k0 = src_block * Q5_K_SOURCE_WPB;
                let valid = (source_k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                for i in 0..valid {
                    acc += deq[i] * x[k0 + i];
                }
            }
            y[row] = acc;
        }
        y
    }

    fn packed_row_dot(
        data_out: &[u8],
        scales_out: &[u8],
        source_k: usize,
        source_n: usize,
        dst_blocks_per_k: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; source_n];
        for row in 0..source_n {
            let mut acc = 0.0f32;
            for (kk, xv) in x.iter().enumerate().take(source_k) {
                let block = kk / Q5_K_DECODED_WPB;
                let elem = kk % Q5_K_DECODED_WPB;
                let block_idx = row * dst_blocks_per_k + block;
                let q = data_out[block_idx * Q5_K_DECODED_DATA_BYTES + elem] as f32;

                let s_idx = block_idx * Q5_K_DECODED_SCALE_BYTES;
                let s = f32::from_le_bytes([
                    scales_out[s_idx],
                    scales_out[s_idx + 1],
                    scales_out[s_idx + 2],
                    scales_out[s_idx + 3],
                ]);
                let a = f32::from_le_bytes([
                    scales_out[s_idx + 4],
                    scales_out[s_idx + 5],
                    scales_out[s_idx + 6],
                    scales_out[s_idx + 7],
                ]);
                acc += (s * q + a) * *xv;
            }
            y[row] = acc;
        }
        y
    }

    fn make_random_q5_raw(seed: &mut u32, source_k: usize, source_n: usize) -> Vec<u8> {
        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let mut raw = vec![0u8; source_n * src_blocks_per_k * Q5_K_SOURCE_BLOCK_BYTES];
        for row in 0..source_n {
            for sb in 0..src_blocks_per_k {
                let off = (row * src_blocks_per_k + sb) * Q5_K_SOURCE_BLOCK_BYTES;
                let blk = &mut raw[off..off + Q5_K_SOURCE_BLOCK_BYTES];
                for b in blk.iter_mut() {
                    *b = (next_u32(seed) & 0xFF) as u8;
                }
                let d = 0.01f32 + ((next_u32(seed) & 0xFF) as f32) * (6.0 / 255.0);
                let dmin = 0.01f32 + ((next_u32(seed) & 0xFF) as f32) * (6.0 / 255.0);
                let d_bits = f16::from_f32(d).to_bits();
                let dm_bits = f16::from_f32(dmin).to_bits();
                blk[0] = (d_bits & 0xFF) as u8;
                blk[1] = (d_bits >> 8) as u8;
                blk[2] = (dm_bits & 0xFF) as u8;
                blk[3] = (dm_bits >> 8) as u8;
            }
        }
        raw
    }

    #[test]
    fn q5_k_rowmajor_repack_matches_direct_ggml_matmul_generic() {
        let mut seed = 0xD15C_A11Eu32;
        let source_k = 320usize;
        let source_n = 48usize;
        let raw = make_random_q5_raw(&mut seed, source_k, source_n);

        let mut x = vec![0.0f32; source_k];
        for xv in &mut x {
            let v = (seed as i32 % 2000) as f32 / 1000.0 - 1.0;
            *xv = v;
            seed = seed.rotate_left(7) ^ 0x9E37_79B9;
        }

        let y_ref = direct_ggml_row_dot(&raw, source_k, source_n, &x);
        let (data_out, scales_out, dst_blocks_per_k) = repack_rowmajor_q5(&raw, source_k, source_n);
        let y_packed = packed_row_dot(&data_out, &scales_out, source_k, source_n, dst_blocks_per_k, &x);

        for i in 0..source_n {
            let diff = (y_ref[i] - y_packed[i]).abs();
            assert!(
                diff <= 6.0e-4,
                "row {i} mismatch: ref={} packed={} diff={}",
                y_ref[i],
                y_packed[i],
                diff
            );
        }
    }

    #[test]
    fn q5_k_rowmajor_embedding_swapped_orientation_matches_direct_ggml_matmul() {
        // Simulate token_embd dims = [vocab, d_model] where loader uses source_k=d_model, source_n=vocab.
        let mut seed = 0xE771_BEDD_u32;
        let d_model = 320usize;
        let vocab = 57usize;
        let source_k = d_model;
        let source_n = vocab;

        let raw = make_random_q5_raw(&mut seed, source_k, source_n);
        let mut x = vec![0.0f32; source_k];
        for xv in &mut x {
            let v = (seed as i32 % 2000) as f32 / 1000.0 - 1.0;
            *xv = v;
            seed = seed.rotate_left(9) ^ 0x7F4A_7C15;
        }

        let y_ref = direct_ggml_row_dot(&raw, source_k, source_n, &x);
        let (data_out, scales_out, dst_blocks_per_k) = repack_rowmajor_q5(&raw, source_k, source_n);
        let y_packed = packed_row_dot(&data_out, &scales_out, source_k, source_n, dst_blocks_per_k, &x);

        for i in 0..source_n {
            let diff = (y_ref[i] - y_packed[i]).abs();
            assert!(
                diff <= 6.0e-4,
                "row {i} mismatch: ref={} packed={} diff={}",
                y_ref[i],
                y_packed[i],
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires local Q5_K_M GGUF and Metal device"]
    fn q5_k_real_model_canonical_attn_q_matches_direct_ggml_row_dot() {
        let model_path = std::env::var("METALLIC_Q5K_MODEL").unwrap_or_else(|_| {
            let candidates = [
                "models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&candidates[0])
                .to_string()
        });
        let model = ModelLoader::from_file(&model_path).expect("failed to load Q5 model");

        let tensor_name = "blk.0.attn_q.weight";
        let info = model.tensor_info(tensor_name).expect("missing blk.0.attn_q.weight");
        assert_eq!(info.data_type, Dtype::Q5_K, "expected Q5_K tensor");
        assert_eq!(info.dimensions.len(), 2, "expected 2D tensor");
        let source_k = info.dimensions[0] as usize;
        let source_n = info.dimensions[1] as usize;

        let mut foundry = crate::Foundry::new().expect("foundry init failed");
        let loaded = PolicyQ5K
            .load_weights(
                &mut foundry,
                model.as_ref(),
                tensor_name,
                tensor_name,
                Layout::Canonical {
                    expected_k: source_k,
                    expected_n: source_n,
                },
            )
            .expect("q5 canonical load failed");
        assert_eq!(loaded.len(), 2, "expected weights + scales");

        let weights = &loaded[0].1;
        let scales = &loaded[1].1;
        let weights_bytes = {
            let len = source_n * source_k;
            let ptr = weights.buffer.as_ref().expect("missing weights buffer").contents() as *const u8;
            // SAFETY: buffer points to at least len bytes in shared storage.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let scales_bytes = {
            let len = scales.dims()[0];
            let ptr = scales.buffer.as_ref().expect("missing scales buffer").contents() as *const u8;
            // SAFETY: buffer points to at least len bytes in shared storage.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };

        let raw_guard = model.tensor_data(tensor_name).expect("raw tensor data unavailable");
        let raw = raw_guard.as_slice();
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let dst_blocks_per_k = source_k.div_ceil(Q5_K_DECODED_WPB);

        let mut x = vec![0.0f32; source_k];
        let mut seed = 0x1234_5678u32;
        for v in &mut x {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        }

        for &row in &[0usize, 17usize, 123usize] {
            let mut ref_acc = 0.0f32;
            for src_block in 0..src_blocks_per_k {
                let src_off = (row * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                let deq = dequant_direct_ggml_style(&blk);
                let k0 = src_block * Q5_K_SOURCE_WPB;
                let valid = (source_k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                for i in 0..valid {
                    ref_acc += deq[i] * x[k0 + i];
                }
            }

            let mut packed_acc = 0.0f32;
            for (k, xv) in x.iter().enumerate().take(source_k) {
                let block = k / Q5_K_DECODED_WPB;
                let elem = k % Q5_K_DECODED_WPB;
                let weight_idx = elem + Q5_K_DECODED_WPB * (row + block * source_n);
                let q = weights_bytes[weight_idx] as f32;

                let scale_block_idx = row * dst_blocks_per_k + block;
                let s_idx = scale_block_idx * Q5_K_DECODED_SCALE_BYTES;
                let s = f32::from_le_bytes([
                    scales_bytes[s_idx],
                    scales_bytes[s_idx + 1],
                    scales_bytes[s_idx + 2],
                    scales_bytes[s_idx + 3],
                ]);
                let a = f32::from_le_bytes([
                    scales_bytes[s_idx + 4],
                    scales_bytes[s_idx + 5],
                    scales_bytes[s_idx + 6],
                    scales_bytes[s_idx + 7],
                ]);
                packed_acc += (s * q + a) * *xv;
            }

            let diff = (ref_acc - packed_acc).abs();
            assert!(
                diff <= 4e-3,
                "canonical real-model row {} diff too large: ref={} packed={} diff={}",
                row,
                ref_acc,
                packed_acc,
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires local Q5_K_M GGUF and Metal device"]
    fn q5_k_real_model_embedding_rowmajor_matches_direct_ggml_row_dot() {
        let model_path = std::env::var("METALLIC_Q5K_MODEL").unwrap_or_else(|_| {
            let candidates = [
                "models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&candidates[0])
                .to_string()
        });
        let model = ModelLoader::from_file(&model_path).expect("failed to load Q5 model");

        let tensor_name = "token_embd.weight";
        let info = model.tensor_info(tensor_name).expect("missing token_embd.weight");
        assert_eq!(info.data_type, Dtype::Q5_K, "expected Q5_K embedding");
        assert_eq!(info.dimensions.len(), 2, "expected 2D tensor");
        let source_n = info.dimensions[0] as usize;
        let source_k = info.dimensions[1] as usize;

        let mut foundry = crate::Foundry::new().expect("foundry init failed");
        let loaded = PolicyQ5K
            .load_weights(&mut foundry, model.as_ref(), tensor_name, "embedding", Layout::RowMajor)
            .expect("q5 rowmajor embedding load failed");
        assert_eq!(loaded.len(), 2, "expected weights + scales");

        let weights = &loaded[0].1;
        let scales = &loaded[1].1;
        let weights_bytes = {
            let len = source_n * source_k;
            let ptr = weights.buffer.as_ref().expect("missing weights buffer").contents() as *const u8;
            // SAFETY: buffer points to at least len bytes in shared storage.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let scales_bytes = {
            let len = scales.dims()[0];
            let ptr = scales.buffer.as_ref().expect("missing scales buffer").contents() as *const u8;
            // SAFETY: buffer points to at least len bytes in shared storage.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };

        let raw_guard = model.tensor_data(tensor_name).expect("raw embedding data unavailable");
        let raw = raw_guard.as_slice();
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let dst_blocks_per_k = source_k.div_ceil(Q5_K_DECODED_WPB);

        let mut x = vec![0.0f32; source_k];
        let mut seed = 0x9E37_79B9u32;
        for v in &mut x {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        }

        for &row in &[0usize, 7usize, 1024usize] {
            let mut ref_acc = 0.0f32;
            for src_block in 0..src_blocks_per_k {
                let src_off = (row * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                let deq = dequant_direct_ggml_style(&blk);
                let k0 = src_block * Q5_K_SOURCE_WPB;
                let valid = (source_k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                for i in 0..valid {
                    ref_acc += deq[i] * x[k0 + i];
                }
            }

            let mut packed_acc = 0.0f32;
            for (k, xv) in x.iter().enumerate().take(source_k) {
                let block = k / Q5_K_DECODED_WPB;
                let elem = k % Q5_K_DECODED_WPB;
                let block_idx = row * dst_blocks_per_k + block;
                let q = weights_bytes[block_idx * Q5_K_DECODED_DATA_BYTES + elem] as f32;

                let s_idx = block_idx * Q5_K_DECODED_SCALE_BYTES;
                let s = f32::from_le_bytes([
                    scales_bytes[s_idx],
                    scales_bytes[s_idx + 1],
                    scales_bytes[s_idx + 2],
                    scales_bytes[s_idx + 3],
                ]);
                let a = f32::from_le_bytes([
                    scales_bytes[s_idx + 4],
                    scales_bytes[s_idx + 5],
                    scales_bytes[s_idx + 6],
                    scales_bytes[s_idx + 7],
                ]);
                packed_acc += (s * q + a) * *xv;
            }

            let diff = (ref_acc - packed_acc).abs();
            assert!(
                diff <= 4e-3,
                "embedding real-model row {} diff too large: ref={} packed={} diff={}",
                row,
                ref_acc,
                packed_acc,
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires local Q5_K_M GGUF and Metal device"]
    fn q5_k_real_model_gemv_kernel_matches_direct_ggml() {
        let model_path = std::env::var("METALLIC_Q5K_MODEL").unwrap_or_else(|_| {
            let candidates = [
                "models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&candidates[0])
                .to_string()
        });
        let model = ModelLoader::from_file(&model_path).expect("failed to load Q5 model");
        let tensor_name = "blk.0.attn_q.weight";
        let info = model.tensor_info(tensor_name).expect("missing blk.0.attn_q.weight");
        assert_eq!(info.data_type, Dtype::Q5_K, "expected Q5_K tensor");
        let source_k = info.dimensions[0] as usize;
        let source_n = info.dimensions[1] as usize;
        let n_dim = source_n;

        let mut foundry = crate::Foundry::new().expect("foundry init failed");
        let loaded = PolicyQ5K
            .load_weights(
                &mut foundry,
                model.as_ref(),
                tensor_name,
                tensor_name,
                Layout::Canonical {
                    expected_k: source_k,
                    expected_n: source_n,
                },
            )
            .expect("q5 canonical load failed");
        let weights = loaded[0].1.clone();
        let scales = loaded[1].1.clone();

        let mut x_f32 = vec![0.0f32; source_k];
        let mut seed = 0xA5A5_5A5Au32;
        for v in &mut x_f32 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        }
        let x_f16: Vec<f16> = x_f32.iter().copied().map(f16::from_f32).collect();

        let x_buf = foundry
            .device
            .new_buffer(
                source_k * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc x buffer failed");
        x_buf.write_via_slice(source_k, |dst: &mut [f16]| dst.copy_from_slice(&x_f16));
        let out_buf = foundry
            .device
            .new_buffer(
                n_dim * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc out buffer failed");
        out_buf.write_via_slice(n_dim, |dst: &mut [f16]| {
            for v in dst.iter_mut() {
                *v = f16::from_f32(0.0);
            }
        });

        let input = crate::types::TensorArg::from_buffer(x_buf, Dtype::F16, vec![source_k], vec![1]);
        let output = crate::types::TensorArg::from_buffer(out_buf.clone(), Dtype::F16, vec![n_dim], vec![1]);
        let scale_bytes = scales;
        let weights_arg = weights;

        let kernel = crate::metals::gemv::get_gemv_v2_kernel(
            std::sync::Arc::new(PolicyQ5K),
            Layout::Canonical {
                expected_k: source_k,
                expected_n: source_n,
            },
            crate::metals::gemv::GemvStrategy::Canonical,
            crate::policy::activation::Activation::None,
        );

        let args = crate::metals::gemv::step::GemvV2Args {
            weights: weights_arg,
            scale_bytes,
            input,
            output: output.clone(),
            k_dim: source_k as u32,
            n_dim: n_dim as u32,
            weights_per_block: 16,
            bias: output.clone(),
            has_bias: 0,
            alpha: 1.0,
            residual: output.clone(),
            has_residual: 0,
            beta: 0.0,
        };
        let dispatch = crate::types::DispatchConfig::warp_per_row(n_dim as u32, 1);
        foundry.run(&kernel.bind_arc(args, dispatch)).expect("gemv run failed");
        foundry.synchronize().expect("synchronize failed");

        let y_gpu: Vec<f32> = unsafe {
            let ptr = out_buf.contents() as *const f16;
            let slice = std::slice::from_raw_parts(ptr, n_dim);
            slice.iter().map(|v| v.to_f32()).collect()
        };

        let raw_guard = model.tensor_data(tensor_name).expect("raw tensor data unavailable");
        let raw = raw_guard.as_slice();
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let compare_rows = 256usize.min(n_dim);
        let mut y_ref = vec![0.0f32; compare_rows];
        for row in 0..compare_rows {
            let mut acc = 0.0f32;
            for src_block in 0..src_blocks_per_k {
                let src_off = (row * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                let deq = dequant_direct_ggml_style(&blk);
                let k0 = src_block * Q5_K_SOURCE_WPB;
                let valid = (source_k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                for i in 0..valid {
                    acc += deq[i] * x_f32[k0 + i];
                }
            }
            y_ref[row] = acc;
        }

        for i in 0..compare_rows {
            let diff = (y_ref[i] - y_gpu[i]).abs();
            assert!(
                diff <= 8e-2,
                "gemv q5 mismatch row {}: ref={} gpu={} diff={}",
                i,
                y_ref[i],
                y_gpu[i],
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires local Q5_K_M GGUF and Metal device"]
    fn q5_k_real_model_gemm_kernel_matches_direct_ggml() {
        let model_path = std::env::var("METALLIC_Q5K_MODEL").unwrap_or_else(|_| {
            let candidates = [
                "models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&candidates[0])
                .to_string()
        });
        let model = ModelLoader::from_file(&model_path).expect("failed to load Q5 model");
        let tensor_name = "blk.0.attn_q.weight";
        let info = model.tensor_info(tensor_name).expect("missing blk.0.attn_q.weight");
        assert_eq!(info.data_type, Dtype::Q5_K, "expected Q5_K tensor");
        let k_dim = info.dimensions[0] as usize;
        let n_dim = info.dimensions[1] as usize;
        let m_dim = 4usize;

        let mut foundry = crate::Foundry::new().expect("foundry init failed");
        let loaded = PolicyQ5K
            .load_weights(
                &mut foundry,
                model.as_ref(),
                &tensor_name,
                &tensor_name,
                Layout::Canonical {
                    expected_k: k_dim,
                    expected_n: n_dim,
                },
            )
            .expect("q5 canonical load failed");
        let b = loaded[0].1.clone();
        let b_scales = loaded[1].1.clone();

        let mut a_f32 = vec![0.0f32; m_dim * k_dim];
        let mut seed = 0xDEAD_BEEFu32;
        for v in &mut a_f32 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed & 0xFFFF) as f32 / 32768.0 - 1.0;
        }
        let a_f16: Vec<f16> = a_f32.iter().copied().map(f16::from_f32).collect();

        let a_buf = foundry
            .device
            .new_buffer(
                a_f16.len() * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc A buffer failed");
        a_buf.write_via_slice(a_f16.len(), |dst: &mut [f16]| dst.copy_from_slice(&a_f16));

        let d_buf = foundry
            .device
            .new_buffer(
                m_dim * n_dim * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc D buffer failed");
        d_buf.write_via_slice(m_dim * n_dim, |dst: &mut [f16]| {
            for v in dst.iter_mut() {
                *v = f16::from_f32(0.0);
            }
        });

        let a = crate::types::TensorArg::from_buffer(a_buf, Dtype::F16, vec![m_dim, k_dim], vec![k_dim, 1]);
        let d = crate::types::TensorArg::from_buffer(d_buf.clone(), Dtype::F16, vec![m_dim, n_dim], vec![n_dim, 1]);

        let config = crate::metals::mma::stages::TileConfig::auto_select(m_dim, n_dim);
        let params = crate::metals::gemm::GemmParams::simple(m_dim as i32, n_dim as i32, k_dim as i32, false, false, config);
        let dispatch = crate::metals::gemm::step::gemm_dispatch_config(&params, config);
        let kernel = crate::metals::gemm::step::get_gemm_kernel(
            std::sync::Arc::new(crate::policy::f16::PolicyF16),
            std::sync::Arc::new(PolicyQ5K),
            false,
            false,
            config,
            false,
            false,
            crate::policy::activation::Activation::None,
        );

        let args = crate::metals::gemm::GemmV2 {
            a,
            b,
            d: d.clone(),
            c: Some(d.clone()),
            bias: Some(d.clone()),
            b_scales: Some(b_scales),
            derived_b_scales: crate::types::TensorArg::default(),
            weights_per_block: 16,
            alpha: 1.0,
            beta: 0.0,
            b_is_canonical: 1,
            params,
            m_dim: crate::spec::DynamicValue::Literal(m_dim as u32),
            n_dim: crate::spec::DynamicValue::Literal(n_dim as u32),
            k_dim: crate::spec::DynamicValue::Literal(k_dim as u32),
            transpose_a: false,
            transpose_b: false,
            tile_config: Some(config),
            activation: crate::policy::activation::Activation::None,
        };
        foundry.run(&kernel.bind_arc(args, dispatch)).expect("gemm run failed");
        foundry.synchronize().expect("synchronize failed");

        let d_gpu: Vec<f32> = unsafe {
            let ptr = d_buf.contents() as *const f16;
            let slice = std::slice::from_raw_parts(ptr, m_dim * n_dim);
            slice.iter().map(|v| v.to_f32()).collect()
        };

        let raw_guard = model.tensor_data(tensor_name).expect("raw tensor data unavailable");
        let raw = raw_guard.as_slice();
        let src_blocks_per_k = k_dim.div_ceil(Q5_K_SOURCE_WPB);

        let compare_rows_n = 64usize.min(n_dim);
        for m in 0..m_dim {
            for n in 0..compare_rows_n {
                let mut acc = 0.0f32;
                for src_block in 0..src_blocks_per_k {
                    let src_off = (n * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                    let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                    blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                    let deq = dequant_direct_ggml_style(&blk);
                    let k0 = src_block * Q5_K_SOURCE_WPB;
                    let valid = (k_dim.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                    for i in 0..valid {
                        acc += deq[i] * a_f16[m * k_dim + k0 + i].to_f32();
                    }
                }
                let got = d_gpu[m * n_dim + n];
                let diff = (acc - got).abs();
                assert!(
                    diff <= 1.2e-1,
                    "gemm q5 mismatch m={} n={}: ref={} gpu={} diff={}",
                    m,
                    n,
                    acc,
                    got,
                    diff
                );
            }
        }
    }

    #[test]
    #[ignore = "requires local Q5_K_M GGUF and Metal device"]
    fn q5_k_real_model_batched_gemv_kernel_matches_direct_ggml() {
        let model_path = std::env::var("METALLIC_Q5K_MODEL").unwrap_or_else(|_| {
            let candidates = [
                "models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
                "../../../models/Llama-3.3-8B-Instruct-128K.Q5_K_M.gguf",
            ];
            candidates
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&candidates[0])
                .to_string()
        });
        let model = ModelLoader::from_file(&model_path).expect("failed to load Q5 model");
        let tensor_name = "blk.0.attn_q.weight";
        let info = model.tensor_info(tensor_name).expect("missing blk.0.attn_q.weight");
        assert_eq!(info.data_type, Dtype::Q5_K, "expected Q5_K tensor");
        let source_k = info.dimensions[0] as usize;
        let source_n = info.dimensions[1] as usize;
        let batch = 4usize;
        let n_dim = source_n;

        let mut foundry = crate::Foundry::new().expect("foundry init failed");
        let loaded = PolicyQ5K
            .load_weights(
                &mut foundry,
                model.as_ref(),
                tensor_name,
                tensor_name,
                Layout::Canonical {
                    expected_k: source_k,
                    expected_n: source_n,
                },
            )
            .expect("q5 canonical load failed");
        let weights_arg = loaded[0].1.clone();
        let scale_bytes = loaded[1].1.clone();

        let mut x_f16 = vec![f16::from_f32(0.0); batch * source_k];
        let mut seed = 0x1357_9BDFu32;
        for v in &mut x_f16 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = f16::from_f32((seed & 0xFFFF) as f32 / 32768.0 - 1.0);
        }

        let x_buf = foundry
            .device
            .new_buffer(
                x_f16.len() * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc x buffer failed");
        x_buf.write_via_slice(x_f16.len(), |dst: &mut [f16]| dst.copy_from_slice(&x_f16));

        let out_buf = foundry
            .device
            .new_buffer(
                batch * n_dim * std::mem::size_of::<f16>(),
                crate::types::MetalResourceOptions::StorageModeShared,
            )
            .expect("alloc out buffer failed");
        out_buf.write_via_slice(batch * n_dim, |dst: &mut [f16]| {
            for v in dst.iter_mut() {
                *v = f16::from_f32(0.0);
            }
        });

        let input = crate::types::TensorArg::from_buffer(x_buf, Dtype::F16, vec![batch, source_k], vec![source_k, 1]);
        let output = crate::types::TensorArg::from_buffer(out_buf.clone(), Dtype::F16, vec![batch, n_dim], vec![n_dim, 1]);

        let kernel = crate::metals::gemv::get_gemv_v2_kernel(
            std::sync::Arc::new(PolicyQ5K),
            Layout::Canonical {
                expected_k: source_k,
                expected_n: source_n,
            },
            crate::metals::gemv::GemvStrategy::Canonical,
            crate::policy::activation::Activation::None,
        );
        let args = crate::metals::gemv::step::GemvV2Args {
            weights: weights_arg,
            scale_bytes,
            input,
            output: output.clone(),
            k_dim: source_k as u32,
            n_dim: n_dim as u32,
            weights_per_block: 16,
            bias: output.clone(),
            has_bias: 0,
            alpha: 1.0,
            residual: output.clone(),
            has_residual: 0,
            beta: 0.0,
        };
        let dispatch = crate::types::DispatchConfig::warp_per_row(n_dim as u32, batch as u32);
        foundry.run(&kernel.bind_arc(args, dispatch)).expect("batched gemv run failed");
        foundry.synchronize().expect("synchronize failed");

        let y_gpu: Vec<f32> = unsafe {
            let ptr = out_buf.contents() as *const f16;
            let slice = std::slice::from_raw_parts(ptr, batch * n_dim);
            slice.iter().map(|v| v.to_f32()).collect()
        };

        let raw_guard = model.tensor_data(tensor_name).expect("raw tensor data unavailable");
        let raw = raw_guard.as_slice();
        let src_blocks_per_k = source_k.div_ceil(Q5_K_SOURCE_WPB);
        let compare_rows_n = 64usize.min(n_dim);
        for b in 0..batch {
            for n in 0..compare_rows_n {
                let mut acc = 0.0f32;
                for src_block in 0..src_blocks_per_k {
                    let src_off = (n * src_blocks_per_k + src_block) * Q5_K_SOURCE_BLOCK_BYTES;
                    let mut blk = [0u8; Q5_K_SOURCE_BLOCK_BYTES];
                    blk.copy_from_slice(&raw[src_off..src_off + Q5_K_SOURCE_BLOCK_BYTES]);
                    let deq = dequant_direct_ggml_style(&blk);
                    let k0 = src_block * Q5_K_SOURCE_WPB;
                    let valid = (source_k.saturating_sub(k0)).min(Q5_K_SOURCE_WPB);
                    for i in 0..valid {
                        let xv = x_f16[b * source_k + k0 + i].to_f32();
                        acc += deq[i] * xv;
                    }
                }
                let got = y_gpu[b * n_dim + n];
                let diff = (acc - got).abs();
                assert!(
                    diff <= 1.2e-1,
                    "batched gemv q5 mismatch b={} n={}: ref={} gpu={} diff={}",
                    b,
                    n,
                    acc,
                    got,
                    diff
                );
            }
        }
    }
}
