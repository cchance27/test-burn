use metallic_loader::LoadedModel;

use crate::{
    Foundry, compound::Layout, tensor::Dtype, types::{MetalResourceOptions, TensorArg}
};

#[inline]
pub(crate) fn canonical_dst_block_idx(src_block_idx: usize, blocks_per_k: usize, target_n: usize) -> usize {
    let row = src_block_idx / blocks_per_k;
    let block = src_block_idx - row * blocks_per_k;
    block * target_n + row
}

#[inline]
pub(crate) fn split_blocks<const BLOCK_BYTES: usize, const SCALE_BYTES: usize, const DATA_BYTES: usize>(
    raw: &[u8],
    blocks_per_k: usize,
    target_n: usize,
    is_canonical: bool,
    data_out: &mut [u8],
    scales_out: &mut [u8],
    mut write_block: impl FnMut(&[u8], &mut [u8]),
) {
    debug_assert_eq!(BLOCK_BYTES, SCALE_BYTES + DATA_BYTES);
    debug_assert_eq!(raw.len() % BLOCK_BYTES, 0);
    debug_assert_eq!(data_out.len() % DATA_BYTES, 0);
    debug_assert_eq!(scales_out.len() % SCALE_BYTES, 0);

    for (src_block_idx, chunk) in raw.chunks_exact(BLOCK_BYTES).enumerate() {
        let scale_dst_block_idx = src_block_idx;
        let weight_dst_block_idx = if is_canonical {
            canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n)
        } else {
            src_block_idx
        };

        let dst_scale = scale_dst_block_idx * SCALE_BYTES;
        scales_out[dst_scale..dst_scale + SCALE_BYTES].copy_from_slice(&chunk[..SCALE_BYTES]);

        let qs = &chunk[SCALE_BYTES..];
        let dst_data = weight_dst_block_idx * DATA_BYTES;
        let block_out = &mut data_out[dst_data..dst_data + DATA_BYTES];
        write_block(qs, block_out);
    }
}

pub(crate) struct LoadedBlockQuant2D {
    pub weights: TensorArg,
    pub scales: TensorArg,
}

#[inline]
pub(crate) fn load_block_quant_2d<const WPB: usize, const BLOCK_BYTES: usize, const SCALE_BYTES: usize, const DATA_BYTES: usize>(
    foundry: &mut Foundry,
    model: &dyn LoadedModel,
    source_tensor_name: &str,
    weight_dtype: Dtype,
    scales_dtype: Dtype,
    layout: Layout,
    write_block: impl FnMut(&[u8], &mut [u8]),
) -> anyhow::Result<LoadedBlockQuant2D> {
    let tensor_info = model
        .tensor_info(source_tensor_name)
        .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in model", source_tensor_name))?;

    let dims: Vec<usize> = tensor_info.dimensions.clone();
    if dims.len() != 2 {
        return Err(anyhow::anyhow!("Quant tensor '{}' must be 2D (got {:?})", source_tensor_name, dims));
    }

    let (target_k, target_n, is_canonical) = match layout {
        Layout::RowMajor => (dims[1], dims[0], false),
        Layout::ColMajor => (dims[0], dims[1], false),
        Layout::Canonical { expected_k, expected_n } => (expected_k, expected_n, true),
    };

    let tensor_data_guard = model.tensor_data(source_tensor_name)?;
    let raw = tensor_data_guard.as_slice();

    if tensor_info.data_type != weight_dtype {
        return Err(anyhow::anyhow!(
            "Tensor '{}' dtype mismatch: expected {:?}, got {:?}",
            source_tensor_name,
            weight_dtype,
            tensor_info.data_type
        ));
    }

    let contiguous_dim_len = if is_canonical {
        if !((dims[0] == target_k && dims[1] == target_n) || (dims[0] == target_n && dims[1] == target_k)) {
            return Err(anyhow::anyhow!(
                "Canonical tensor '{}' dims {:?} mismatch (K,N)=({},{})",
                source_tensor_name,
                dims,
                target_k,
                target_n
            ));
        }
        target_k
    } else {
        dims[1]
    };

    if contiguous_dim_len % WPB != 0 {
        return Err(anyhow::anyhow!(
            "Quant tensor '{}' contig dim {} not divisible by {}",
            source_tensor_name,
            contiguous_dim_len,
            WPB
        ));
    }

    let blocks_per_k = contiguous_dim_len / WPB;
    let n_dim_len = if is_canonical { target_n } else { dims[0] };
    let total_blocks = n_dim_len * blocks_per_k;
    let expected_bytes = total_blocks * BLOCK_BYTES;

    if raw.len() != expected_bytes {
        return Err(anyhow::anyhow!(
            "Quant tensor size mismatch for '{}': got {}, exp {}",
            source_tensor_name,
            raw.len(),
            expected_bytes
        ));
    }

    let data_len = total_blocks * DATA_BYTES;
    let scales_len = total_blocks * SCALE_BYTES;

    let data_buffer = foundry
        .device
        .new_buffer(data_len, MetalResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow::anyhow!("Failed to allocate quant data"))?;

    let scales_buffer = foundry
        .device
        .new_buffer(scales_len, MetalResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow::anyhow!("Failed to allocate quant scales"))?;

    let layout_hint = model
        .metadata()
        .get_string("metallic.gguf.layout_hint")
        .unwrap_or(std::borrow::Cow::Borrowed("nk"));

    if is_canonical && layout_hint != "nk" {
        return Err(anyhow::anyhow!("Canonical quant load requires Nk layout"));
    }

    data_buffer.write_via_slice(data_len, |data_out| {
        scales_buffer.write_via_slice(scales_len, |scales_out| {
            split_blocks::<BLOCK_BYTES, SCALE_BYTES, DATA_BYTES>(
                raw,
                blocks_per_k,
                target_n,
                is_canonical,
                data_out,
                scales_out,
                write_block,
            );
        });
    });

    let weights = TensorArg::from_buffer(
        data_buffer,
        weight_dtype,
        if is_canonical { vec![total_blocks * WPB] } else { dims.clone() },
        if is_canonical {
            vec![1]
        } else {
            crate::tensor::compute_strides(&dims)
        },
    );

    let scales = TensorArg::from_buffer(scales_buffer, scales_dtype, vec![scales_len], vec![1]);

    Ok(LoadedBlockQuant2D { weights, scales })
}
