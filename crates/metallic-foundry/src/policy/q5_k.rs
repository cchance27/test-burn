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

#[path = "q5_k.test.rs"]
mod tests;
