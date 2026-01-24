use metallic_macros::MetalPolicy;

use super::{LoaderStage, MetalPolicyRuntime};
use crate::{
    Foundry, compound::Layout, gguf::{file::GGUFDataType, model_loader::GGUFModel, tensor_info::GGUFRawTensor}, spec::{FastBindings, ResolvedSymbols}, tensor::Dtype, types::{MetalResourceOptions, TensorArg}
};

const Q8_0_WPB: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34; // 2 bytes scale + 32 bytes data
const Q8_0_SCALE_BYTES: usize = 2;

#[inline]
fn q8_0_canonical_dst_block_idx(src_block_idx: usize, blocks_per_k: usize, target_n: usize) -> usize {
    let row = src_block_idx / blocks_per_k;
    let block = src_block_idx - row * blocks_per_k;
    block * target_n + row
}

fn split_q8_0_blocks(raw: &[u8], blocks_per_k: usize, target_n: usize, is_canonical: bool, data_out: &mut [u8], scales_out: &mut [u8]) {
    debug_assert_eq!(raw.len() % Q8_0_BLOCK_BYTES, 0);
    debug_assert_eq!(data_out.len() % Q8_0_WPB, 0);
    debug_assert_eq!(scales_out.len() % Q8_0_SCALE_BYTES, 0);

    for (src_block_idx, chunk) in raw.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        // Scales are always indexed in row-major block order (row * blocks_per_k + block),
        // even when weights are stored in canonical (block-major) order.
        let scale_dst_block_idx = src_block_idx;
        let weight_dst_block_idx = if is_canonical {
            q8_0_canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n)
        } else {
            src_block_idx
        };

        let dst_scale = scale_dst_block_idx * Q8_0_SCALE_BYTES;
        scales_out[dst_scale..dst_scale + Q8_0_SCALE_BYTES].copy_from_slice(&chunk[..Q8_0_SCALE_BYTES]);

        let dst_data = weight_dst_block_idx * Q8_0_WPB;
        data_out[dst_data..dst_data + Q8_0_WPB].copy_from_slice(&chunk[Q8_0_SCALE_BYTES..Q8_0_SCALE_BYTES + Q8_0_WPB]);
    }
}

#[derive(Default, Debug, Clone, MetalPolicy)]
#[policy(
    header = "policies/policy_q8.metal",
    struct_name = "PolicyQ8",
    short_name = "q8",
    element_size = 1,
    block_size = 32,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32
)]
pub struct PolicyQ8;

impl LoaderStage for PolicyQ8 {
    fn params_struct(&self) -> String {
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        // Q8 expects [weight, scales]
        // We panic here if bindings are missing because bind assumes indices are valid
        // and presence was validated during compile/load (or we accept the panic as "unbound tensor").
        // For performance, we can unwrap or use expect.
        let weight = fast_bindings.get(resolved.weights).expect("Q8 weight bound");

        let scales_idx = resolved.scales.expect("Q8 requires scales index");
        let scales = fast_bindings.get(scales_idx).expect("Q8 scales bound");

        smallvec![weight.clone(), scales.clone()]
    }

    fn quantization_type(&self) -> std::sync::Arc<dyn super::MetalPolicyRuntime> {
        std::sync::Arc::new(PolicyQ8)
    }
}

impl MetalPolicyRuntime for PolicyQ8 {
    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(self.clone())
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        gguf: &GGUFModel,
        gguf_tensor_name: &str,
        logical_name: &str,
        layout: Layout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let tensor_info = gguf
            .get_tensor(gguf_tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in GGUF", gguf_tensor_name))?;

        let dims: Vec<usize> = tensor_info.dims().to_vec();
        if dims.len() != 2 {
            return Err(anyhow::anyhow!("Q8_0 tensor '{}' must be 2D (got {:?})", gguf_tensor_name, dims));
        }

        let (target_k, target_n, is_canonical) = match layout {
            Layout::RowMajor => (dims[1], dims[0], false),
            Layout::ColMajor => (dims[0], dims[1], false),
            Layout::Canonical { expected_k, expected_n } => (expected_k, expected_n, true),
        };

        let view = tensor_info.raw_view(&gguf.gguf_file)?;
        let raw = match view {
            GGUFRawTensor::Bytes(b, GGUFDataType::Q8_0) => b,
            _ => {
                return Err(anyhow::anyhow!(
                    "Tensor '{}' is not Q8_0 bytes. Got {:?}",
                    gguf_tensor_name,
                    tensor_info.data_type()
                ));
            }
        };

        let contiguous_dim_len = if is_canonical {
            if !((dims[0] == target_k && dims[1] == target_n) || (dims[0] == target_n && dims[1] == target_k)) {
                return Err(anyhow::anyhow!(
                    "Q8 canonical tensor '{}' dims {:?} mismatch (K,N)=({},{})",
                    gguf_tensor_name,
                    dims,
                    target_k,
                    target_n
                ));
            }
            target_k
        } else {
            dims[1]
        };

        if contiguous_dim_len % Q8_0_WPB != 0 {
            return Err(anyhow::anyhow!(
                "Q8_0 tensor '{}' contig dim {} not divisible by 32",
                gguf_tensor_name,
                contiguous_dim_len
            ));
        }

        // Block count for contiguous dim
        let blocks_per_k = contiguous_dim_len / Q8_0_WPB;
        let n_dim_len = if is_canonical { target_n } else { dims[0] };

        let total_blocks = n_dim_len * blocks_per_k;
        let expected_bytes = total_blocks * Q8_0_BLOCK_BYTES;

        if raw.len() != expected_bytes {
            return Err(anyhow::anyhow!(
                "Q8 tensor size mismatch for '{}': got {}, exp {}",
                gguf_tensor_name,
                raw.len(),
                expected_bytes
            ));
        }

        // Allocate separate Metal buffers
        let data_len = total_blocks * Q8_0_WPB;
        let scales_len = total_blocks * Q8_0_SCALE_BYTES;

        let data_buffer = foundry
            .device
            .new_buffer(data_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q8 data"))?;

        let scales_buffer = foundry
            .device
            .new_buffer(scales_len, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q8 scales"))?;

        if is_canonical {
            if gguf.layout_hint() != crate::gguf::model_loader::GGUFLayoutHint::Nk {
                return Err(anyhow::anyhow!("Q8 canonical requires Nk layout"));
            }
        } else {
            // Standard split keeps the row-major block order.
        }

        // Split blocks into separate {data, scales} buffers (and reorder if canonical).
        data_buffer.write_via_slice(data_len, |data_out| {
            scales_buffer.write_via_slice(scales_len, |scales_out| {
                split_q8_0_blocks(raw, blocks_per_k, target_n, is_canonical, data_out, scales_out);
            });
        });

        let data_arg = TensorArg::from_buffer(
            data_buffer,
            Dtype::U8,
            if is_canonical { vec![data_len] } else { dims.clone() },
            if is_canonical {
                vec![1]
            } else {
                crate::tensor::compute_strides(&dims)
            },
        );

        let scales_arg = TensorArg::from_buffer(scales_buffer, Dtype::U8, vec![scales_len], vec![1]);

        Ok(vec![
            (logical_name.to_string(), data_arg),
            (format!("{}_scales", logical_name), scales_arg),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_canonical_reorders_scales_and_data_consistently() {
        let target_n = 3;
        let blocks_per_k = 2;
        let total_blocks = target_n * blocks_per_k;

        let mut raw = vec![0u8; total_blocks * Q8_0_BLOCK_BYTES];
        for src_block_idx in 0..total_blocks {
            let start = src_block_idx * Q8_0_BLOCK_BYTES;
            raw[start] = src_block_idx as u8; // scale byte 0
            raw[start + 1] = 0xEE; // scale byte 1 sentinel
            raw[start + Q8_0_SCALE_BYTES..start + Q8_0_BLOCK_BYTES].fill((0x10 + src_block_idx) as u8);
        }

        let mut data_out = vec![0u8; total_blocks * Q8_0_WPB];
        let mut scales_out = vec![0u8; total_blocks * Q8_0_SCALE_BYTES];

        split_q8_0_blocks(&raw, blocks_per_k, target_n, true, &mut data_out, &mut scales_out);

        for src_block_idx in 0..total_blocks {
            let scale_start = src_block_idx * Q8_0_SCALE_BYTES;
            assert_eq!(scales_out[scale_start], src_block_idx as u8);
            assert_eq!(scales_out[scale_start + 1], 0xEE);

            let dst = q8_0_canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n);
            let data_start = dst * Q8_0_WPB;
            assert_eq!(data_out[data_start], (0x10 + src_block_idx) as u8);
            assert_eq!(data_out[data_start + Q8_0_WPB - 1], (0x10 + src_block_idx) as u8);
        }
    }
}
