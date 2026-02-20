use metallic_loader::LoadedModel;

use crate::{
    Foundry, compound::Layout, tensor::Dtype, types::{MetalResourceOptions, TensorArg}
};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackedAxis {
    Dim0,
    Dim1,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScaleModel {
    ScaleOnly,
    Affine, // DEBT: add affine plane/mins output when _1 policies are implemented.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockQuantLayout {
    pub packed_axis: PackedAxis,
    pub scale_model: ScaleModel,
}

impl BlockQuantLayout {
    pub const fn standard() -> Self {
        Self {
            packed_axis: PackedAxis::Dim1,
            scale_model: ScaleModel::ScaleOnly,
        }
    }
}

pub(crate) trait BlockQuantCodec {
    const SOURCE_DTYPE: Dtype;
    const SCALES_DTYPE: Dtype;
    const WEIGHTS_PER_BLOCK: usize;
    const BLOCK_BYTES: usize;
    const SCALE_BYTES: usize;
    const DATA_BYTES: usize;
    const LAYOUT: BlockQuantLayout = BlockQuantLayout::standard();

    fn write_block(qs: &[u8], out: &mut [u8]);
}

#[inline]
pub(crate) fn canonical_dst_block_idx(src_block_idx: usize, blocks_per_k: usize, target_n: usize) -> usize {
    let row = src_block_idx / blocks_per_k;
    let block = src_block_idx - row * blocks_per_k;
    block * target_n + row
}

#[inline]
#[allow(dead_code)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockQuantRuntimeSpec {
    pub weight_dtype: Dtype,
    pub scales_dtype: Dtype,
    pub weights_per_block: usize,
    pub block_bytes: usize,
    pub scale_bytes: usize,
    pub data_bytes: usize,
    pub layout: BlockQuantLayout,
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_blocks_runtime(
    raw: &[u8],
    blocks_per_k: usize,
    target_n: usize,
    is_canonical: bool,
    scale_bytes: usize,
    data_bytes: usize,
    block_bytes: usize,
    data_out: &mut [u8],
    scales_out: &mut [u8],
    mut write_block: impl FnMut(&[u8], &mut [u8]),
) {
    debug_assert_eq!(block_bytes, scale_bytes + data_bytes);
    debug_assert_eq!(raw.len() % block_bytes, 0);
    debug_assert_eq!(data_out.len() % data_bytes, 0);
    debug_assert_eq!(scales_out.len() % scale_bytes, 0);

    for (src_block_idx, chunk) in raw.chunks_exact(block_bytes).enumerate() {
        let scale_dst_block_idx = src_block_idx;
        let weight_dst_block_idx = if is_canonical {
            canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n)
        } else {
            src_block_idx
        };

        let dst_scale = scale_dst_block_idx * scale_bytes;
        scales_out[dst_scale..dst_scale + scale_bytes].copy_from_slice(&chunk[..scale_bytes]);

        let qs = &chunk[scale_bytes..];
        let dst_data = weight_dst_block_idx * data_bytes;
        let block_out = &mut data_out[dst_data..dst_data + data_bytes];
        write_block(qs, block_out);
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn load_block_quant_2d<const WPB: usize, const BLOCK_BYTES: usize, const SCALE_BYTES: usize, const DATA_BYTES: usize>(
    foundry: &mut Foundry,
    model: &dyn LoadedModel,
    source_tensor_name: &str,
    weight_dtype: Dtype,
    scales_dtype: Dtype,
    layout: Layout,
    write_block: impl FnMut(&[u8], &mut [u8]),
) -> anyhow::Result<LoadedBlockQuant2D> {
    load_block_quant_2d_runtime(
        foundry,
        model,
        source_tensor_name,
        BlockQuantRuntimeSpec {
            weight_dtype,
            scales_dtype,
            weights_per_block: WPB,
            block_bytes: BLOCK_BYTES,
            scale_bytes: SCALE_BYTES,
            data_bytes: DATA_BYTES,
            layout: BlockQuantLayout::standard(),
        },
        layout,
        write_block,
    )
}

#[inline]
pub(crate) fn load_block_quant_2d_with_codec<C: BlockQuantCodec>(
    foundry: &mut Foundry,
    model: &dyn LoadedModel,
    source_tensor_name: &str,
    layout: Layout,
) -> anyhow::Result<LoadedBlockQuant2D> {
    load_block_quant_2d_runtime(
        foundry,
        model,
        source_tensor_name,
        BlockQuantRuntimeSpec {
            weight_dtype: C::SOURCE_DTYPE,
            scales_dtype: C::SCALES_DTYPE,
            weights_per_block: C::WEIGHTS_PER_BLOCK,
            block_bytes: C::BLOCK_BYTES,
            scale_bytes: C::SCALE_BYTES,
            data_bytes: C::DATA_BYTES,
            layout: C::LAYOUT,
        },
        layout,
        C::write_block,
    )
}

#[inline]
pub(crate) fn load_block_quant_2d_runtime(
    foundry: &mut Foundry,
    model: &dyn LoadedModel,
    source_tensor_name: &str,
    spec: BlockQuantRuntimeSpec,
    layout: Layout,
    write_block: impl FnMut(&[u8], &mut [u8]),
) -> anyhow::Result<LoadedBlockQuant2D> {
    if spec.layout.scale_model != ScaleModel::ScaleOnly {
        return Err(anyhow::anyhow!(
            "Scale model {:?} is not wired yet for '{}' (need affine plane support)",
            spec.layout.scale_model,
            source_tensor_name
        ));
    }

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

    if tensor_info.data_type != spec.weight_dtype {
        return Err(anyhow::anyhow!(
            "Tensor '{}' dtype mismatch: expected {:?}, got {:?}",
            source_tensor_name,
            spec.weight_dtype,
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
        match spec.layout.packed_axis {
            PackedAxis::Dim1 => dims[1],
            PackedAxis::Dim0 => dims[0],
        }
    };

    if contiguous_dim_len % spec.weights_per_block != 0 {
        return Err(anyhow::anyhow!(
            "Quant tensor '{}' contig dim {} not divisible by {}",
            source_tensor_name,
            contiguous_dim_len,
            spec.weights_per_block
        ));
    }

    let blocks_per_k = contiguous_dim_len / spec.weights_per_block;
    let n_dim_len = if is_canonical {
        target_n
    } else {
        match spec.layout.packed_axis {
            PackedAxis::Dim1 => dims[0],
            PackedAxis::Dim0 => dims[1],
        }
    };
    let total_blocks = n_dim_len * blocks_per_k;
    let expected_bytes = total_blocks * spec.block_bytes;

    if raw.len() != expected_bytes {
        return Err(anyhow::anyhow!(
            "Quant tensor size mismatch for '{}': got {}, exp {}",
            source_tensor_name,
            raw.len(),
            expected_bytes
        ));
    }

    let data_len = total_blocks * spec.data_bytes;
    let scales_len = total_blocks * spec.scale_bytes;

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
            split_blocks_runtime(
                raw,
                blocks_per_k,
                target_n,
                is_canonical,
                spec.scale_bytes,
                spec.data_bytes,
                spec.block_bytes,
                data_out,
                scales_out,
                write_block,
            );
        });
    });

    let weights = TensorArg::from_buffer(
        data_buffer,
        spec.weight_dtype,
        if is_canonical {
            vec![total_blocks * spec.weights_per_block]
        } else {
            dims.clone()
        },
        if is_canonical {
            vec![1]
        } else {
            crate::tensor::compute_strides(&dims)
        },
    );

    let scales = TensorArg::from_buffer(scales_buffer, spec.scales_dtype, vec![scales_len], vec![1]);

    Ok(LoadedBlockQuant2D { weights, scales })
}
