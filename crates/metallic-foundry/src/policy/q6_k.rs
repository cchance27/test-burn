use half::f16;
use metallic_loader::{LoadedModel, quant_spec::Q6_K_SPEC};
use metallic_macros::MetalPolicy;

use super::{LoaderStage, MetalPolicyRuntime};
use crate::{
    Foundry, compound::Layout, policy::block_quant::canonical_dst_block_idx, spec::{FastBindings, ResolvedSymbols}, tensor::Dtype, types::{MetalResourceOptions, TensorArg}
};

const Q6_K_SOURCE_WPB: usize = Q6_K_SPEC.weights_per_block;
const Q6_K_SOURCE_BLOCK_BYTES: usize = Q6_K_SPEC.block_bytes; // ql[128] + qh[64] + scales[i8;16] + d(f16)
const Q6_K_DECODED_WPB: usize = 16;
const Q6_K_DECODED_DATA_BYTES: usize = 16; // i8[16]
const Q6_K_DECODED_SCALE_BYTES: usize = 2; // f16(d * scale)

#[inline]
fn decode_q6_k_block(raw_block: &[u8], q_out: &mut [u8; Q6_K_SOURCE_WPB], scales_out: &mut [u16; Q6_K_SOURCE_WPB / Q6_K_DECODED_WPB]) {
    debug_assert_eq!(raw_block.len(), Q6_K_SOURCE_BLOCK_BYTES);

    // GGML block_q6_K layout is ql, qh, scales, d (d is stored at the tail).
    let d_off = Q6_K_SOURCE_BLOCK_BYTES - 2;
    let d_bits = u16::from(raw_block[d_off]) | (u16::from(raw_block[d_off + 1]) << 8);
    let d = f16::from_bits(d_bits).to_f32();
    let ql = &raw_block[0..128];
    let qh = &raw_block[128..192];
    let sc = &raw_block[192..208];

    for group in 0..16 {
        let scale = (sc[group] as i8) as f32;
        let sd = f16::from_f32(d * scale).to_bits();
        scales_out[group] = sd;
    }

    // Mirrors ggml's dequantize_row_q6_K indexing, but keeps q-values unscaled.
    for n in 0..2 {
        let ql_n = &ql[n * 64..n * 64 + 64];
        let qh_n = &qh[n * 32..n * 32 + 32];
        for l in 0..32 {
            let base = n * 128;
            let q1 = ((ql_n[l] & 0x0F) | ((qh_n[l] & 0x03) << 4)) as i8 - 32;
            let q2 = ((ql_n[l + 32] & 0x0F) | (((qh_n[l] >> 2) & 0x03) << 4)) as i8 - 32;
            let q3 = ((ql_n[l] >> 4) | (((qh_n[l] >> 4) & 0x03) << 4)) as i8 - 32;
            let q4 = ((ql_n[l + 32] >> 4) | (((qh_n[l] >> 6) & 0x03) << 4)) as i8 - 32;

            q_out[base + l] = q1 as u8;
            q_out[base + l + 32] = q2 as u8;
            q_out[base + l + 64] = q3 as u8;
            q_out[base + l + 96] = q4 as u8;
        }
    }
}

#[derive(Debug, Clone, Default, MetalPolicy)]
#[policy(
    header = "policies/policy_q6_k.metal",
    struct_name = "PolicyQ6K",
    short_name = "q6_k",
    element_size = 1,
    block_size = 16,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 18,
    weights_per_block = 16
)]
pub struct PolicyQ6K;

impl LoaderStage for PolicyQ6K {
    fn params_struct(&self) -> String {
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        let weight = fast_bindings.get(resolved.weights).expect("Q6_K weight bound");
        let scales_idx = resolved.scales.expect("Q6_K requires scales index");
        let scales = fast_bindings.get(scales_idx).expect("Q6_K scales bound");
        smallvec![weight.clone(), scales.clone()]
    }

    fn quantization_type(&self) -> std::sync::Arc<dyn MetalPolicyRuntime> {
        std::sync::Arc::new(PolicyQ6K)
    }
}

impl MetalPolicyRuntime for PolicyQ6K {
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
            return Err(anyhow::anyhow!("Q6_K tensor '{}' must be 2D (got {:?})", source_tensor_name, dims));
        }

        let (target_k, target_n, is_canonical) = match layout {
            Layout::RowMajor => (dims[1], dims[0], false),
            Layout::ColMajor => (dims[0], dims[1], false),
            Layout::Canonical { expected_k, expected_n } => (expected_k, expected_n, true),
        };

        let tensor_data_guard = model.tensor_data(source_tensor_name)?;
        let raw = tensor_data_guard.as_slice();

        if tensor_info.data_type != Dtype::Q6_K {
            return Err(anyhow::anyhow!(
                "Tensor '{}' dtype mismatch: expected {:?}, got {:?}",
                source_tensor_name,
                Dtype::Q6_K,
                tensor_info.data_type
            ));
        }

        let (source_k_len, source_n_len) = if is_canonical {
            if !((dims[0] == target_k && dims[1] == target_n) || (dims[0] == target_n && dims[1] == target_k)) {
                return Err(anyhow::anyhow!(
                    "Canonical Q6_K tensor '{}' dims {:?} mismatch (K,N)=({},{})",
                    source_tensor_name,
                    dims,
                    target_k,
                    target_n
                ));
            }
            (target_k, target_n)
        } else {
            // GGUF quant blocks are packed along ne[0] (K axis), with rows over ne[1].
            // For RowMajor in Foundry this means source dims are typically [K, N].
            match layout {
                Layout::RowMajor => (dims[0], dims[1]),
                Layout::ColMajor => (dims[1], dims[0]),
                Layout::Canonical { .. } => unreachable!("handled above"),
            }
        };

        if raw.len() % Q6_K_SOURCE_BLOCK_BYTES != 0 {
            return Err(anyhow::anyhow!(
                "Q6_K tensor '{}' byte size {} is not a multiple of block bytes {}",
                source_tensor_name,
                raw.len(),
                Q6_K_SOURCE_BLOCK_BYTES
            ));
        }

        if source_n_len == 0 {
            return Err(anyhow::anyhow!("Q6_K tensor '{}' has zero rows", source_tensor_name));
        }

        let total_src_blocks = raw.len() / Q6_K_SOURCE_BLOCK_BYTES;
        if !total_src_blocks.is_multiple_of(source_n_len) {
            return Err(anyhow::anyhow!(
                "Q6_K tensor '{}' source blocks {} not divisible by rows {}",
                source_tensor_name,
                total_src_blocks,
                source_n_len
            ));
        }

        let src_blocks_per_k = total_src_blocks / source_n_len;
        let expected_src_blocks_per_k = source_k_len.div_ceil(Q6_K_SOURCE_WPB);
        if src_blocks_per_k != expected_src_blocks_per_k {
            return Err(anyhow::anyhow!(
                "Q6_K tensor '{}' blocks-per-row mismatch: got {}, expected {} from K={}",
                source_tensor_name,
                src_blocks_per_k,
                expected_src_blocks_per_k,
                source_k_len
            ));
        }

        let dst_blocks_per_k = source_k_len.div_ceil(Q6_K_DECODED_WPB);
        if src_blocks_per_k * (Q6_K_SOURCE_WPB / Q6_K_DECODED_WPB) < dst_blocks_per_k {
            return Err(anyhow::anyhow!(
                "Q6_K tensor '{}' has insufficient source blocks: src_blocks_per_k={}, dst_blocks_per_k={}",
                source_tensor_name,
                src_blocks_per_k,
                dst_blocks_per_k
            ));
        }
        let n_dim_len = source_n_len;

        let total_dst_blocks = n_dim_len * dst_blocks_per_k;
        let data_len = total_dst_blocks * Q6_K_DECODED_DATA_BYTES;
        let scales_len = total_dst_blocks * Q6_K_DECODED_SCALE_BYTES;

        let data_buffer = foundry
            .device
            .new_buffer(data_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q6_K data"))?;
        let scales_buffer = foundry
            .device
            .new_buffer(scales_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q6_K scales"))?;

        let layout_hint = model
            .metadata()
            .get_string("metallic.gguf.layout_hint")
            .unwrap_or(std::borrow::Cow::Borrowed("nk"));
        if is_canonical && layout_hint != "nk" {
            return Err(anyhow::anyhow!("Canonical Q6_K load requires Nk layout"));
        }

        data_buffer.write_via_slice(data_len, |data_out| {
            scales_buffer.write_via_slice(scales_len, |scales_out| {
                let mut q_vals = [0u8; Q6_K_SOURCE_WPB];
                let mut scale_bits = [0u16; Q6_K_SOURCE_WPB / Q6_K_DECODED_WPB];

                for row in 0..n_dim_len {
                    for src_block in 0..src_blocks_per_k {
                        let src_block_idx = row * src_blocks_per_k + src_block;
                        let src_off = src_block_idx * Q6_K_SOURCE_BLOCK_BYTES;
                        let chunk = &raw[src_off..src_off + Q6_K_SOURCE_BLOCK_BYTES];
                        decode_q6_k_block(chunk, &mut q_vals, &mut scale_bits);

                        let valid_values = (source_k_len.saturating_sub(src_block * Q6_K_SOURCE_WPB)).min(Q6_K_SOURCE_WPB);
                        let valid_groups = valid_values.div_ceil(Q6_K_DECODED_WPB);
                        for (g, &bits) in scale_bits.iter().enumerate().take(valid_groups) {
                            let src_block16_idx = row * dst_blocks_per_k + src_block * 16 + g;
                            let dst_weight_block_idx = if is_canonical {
                                canonical_dst_block_idx(src_block16_idx, dst_blocks_per_k, target_n)
                            } else {
                                src_block16_idx
                            };

                            let q_src = g * Q6_K_DECODED_DATA_BYTES;
                            let q_dst = dst_weight_block_idx * Q6_K_DECODED_DATA_BYTES;
                            data_out[q_dst..q_dst + Q6_K_DECODED_DATA_BYTES]
                                .copy_from_slice(&q_vals[q_src..q_src + Q6_K_DECODED_DATA_BYTES]);

                            let s_dst = src_block16_idx * Q6_K_DECODED_SCALE_BYTES;
                            scales_out[s_dst] = (bits & 0xFF) as u8;
                            scales_out[s_dst + 1] = (bits >> 8) as u8;
                        }
                    }
                }
            });
        });

        let weights = TensorArg::from_buffer(
            data_buffer,
            Dtype::Q6_K,
            if is_canonical {
                vec![total_dst_blocks * Q6_K_DECODED_WPB]
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

#[path = "q6_k.test.rs"]
mod tests;
