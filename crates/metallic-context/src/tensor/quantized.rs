use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBlitCommandEncoder as _, MTLDevice, MTLResourceOptions};

use crate::{
    Context, MetalError, operation::CommandBuffer, tensor::{Dtype, RetainedBuffer, Tensor, TensorElement, U8}
};

/// Swizzle NK row-major Q8_0 bytes (rows = output dim, cols = input dim) into a layout where
/// all rows for a given K-block are contiguous, improving streaming efficiency.
/// Returns `None` when the provided dimensions do not match the byte length.
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
#[repr(C, packed)]
pub struct Q8_0Block {
    pub scale_f16_le: [u8; 2],
    pub qs: [i8; 32],
}

pub const Q8_0_BLOCK_SIZE_BYTES: usize = 34;
pub const Q8_0_WEIGHTS_PER_BLOCK: usize = 32;
pub const Q8_0_SCALE_BYTES_PER_BLOCK: usize = 2;

/// A packed Q8_0 tensor stored in canonical split layout on GPU.
/// - `data` holds tightly packed int8 weights (32 values per block)
/// - `scales` holds the corresponding fp16 scale for each block
/// - `logical_dims` is the tensor’s float-shape (elements view), used by kernels
#[derive(Clone)]
pub struct QuantizedQ8_0Tensor {
    pub data: Tensor<U8>,
    pub scales: Tensor<U8>,
    pub logical_dims: Vec<usize>,
    pub blocks_per_k: usize,
}

impl QuantizedQ8_0Tensor {
    #[inline]
    pub fn logical_len(&self) -> usize {
        self.logical_dims.iter().product()
    }

    #[inline]
    pub fn raw_len_bytes(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn blocks(&self) -> usize {
        self.data.len() / Q8_0_WEIGHTS_PER_BLOCK
    }

    #[inline]
    pub fn buffer(&self) -> &RetainedBuffer {
        &self.data.buf
    }

    #[inline]
    pub fn scales_buffer(&self) -> &Tensor<U8> {
        &self.scales
    }

    #[inline]
    pub fn blocks_per_k(&self) -> usize {
        self.blocks_per_k
    }

    #[inline]
    pub fn dtype(&self) -> Dtype {
        Dtype::U8
    }

    pub fn from_split_bytes_in_context<TCtx: TensorElement>(
        logical_dims: Vec<usize>,
        data_bytes: &[u8],
        scale_bytes: &[u8],
        ctx: &Context<TCtx>,
    ) -> Result<Self, MetalError> {
        if !data_bytes.len().is_multiple_of(Q8_0_WEIGHTS_PER_BLOCK) {
            return Err(MetalError::InvalidOperation(format!(
                "Q8_0 split data length {} is not divisible by {}",
                data_bytes.len(),
                Q8_0_WEIGHTS_PER_BLOCK
            )));
        }
        if !scale_bytes.len().is_multiple_of(Q8_0_SCALE_BYTES_PER_BLOCK) {
            return Err(MetalError::InvalidOperation(format!(
                "Q8_0 split scales length {} is not divisible by {}",
                scale_bytes.len(),
                Q8_0_SCALE_BYTES_PER_BLOCK
            )));
        }

        let logical_elems = logical_dims.iter().product::<usize>();
        let total_weights = data_bytes.len();
        if total_weights < logical_elems {
            return Err(MetalError::InvalidOperation(format!(
                "Q8_0 split data holds {} weights but logical dims require {}",
                total_weights, logical_elems
            )));
        }

        let data = upload_u8_bytes(data_bytes, &ctx.device, &ctx.command_queue)?;
        let scales = upload_u8_bytes(scale_bytes, &ctx.device, &ctx.command_queue)?;
        // Infer blocks_per_k robustly by validating against actual buffer sizes.
        // For split canonical layout: scales_count = blocks_per_k * N, data_bytes = blocks_per_k * N * 32.
        // Evaluate both interpretations of dims: [K,N] and [N,K], and choose the one that matches buffers.
        let scales_count = scale_bytes.len() / Q8_0_SCALE_BYTES_PER_BLOCK;
        let blocks_per_k = if logical_dims.len() >= 2 {
            let n0 = logical_dims[0];
            let k0 = logical_dims[1];
            let cand0 = k0.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
            let ok0 = cand0.saturating_mul(n0) == scales_count;

            let k1 = logical_dims[0];
            let n1 = logical_dims[1];
            let cand1 = k1.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
            let ok1 = cand1.saturating_mul(n1) == scales_count;

            if ok0 {
                cand0
            } else if ok1 {
                cand1
            } else {
                k1.div_ceil(Q8_0_WEIGHTS_PER_BLOCK) // Fallback
            }
        } else {
            0
        };
        Ok(Self {
            data,
            scales,
            logical_dims,
            blocks_per_k,
        })
    }
}

/// Canonical quantized tensor representation used by high-performance kernels.
/// Stores tightly packed data and auxiliary buffers (e.g., scales) along with metadata
/// such as blocks_per_k so that kernels can deterministically compute offsets.
#[derive(Clone)]
pub struct CanonicalQuantTensor {
    pub kind: CanonicalQuantKind,
    pub logical_dims: Vec<usize>,
    pub data: Tensor<U8>,
    pub scales: Tensor<U8>,
    pub zero_points: Option<Tensor<U8>>,
    pub blocks_per_k: usize,
    pub weights_per_block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalQuantKind {
    Q8_0,
}

impl CanonicalQuantTensor {
    pub fn from_q8_0_blocks<TCtx: TensorElement>(
        logical_dims: Vec<usize>,
        raw_blocks: &[u8],
        ctx: &Context<TCtx>,
    ) -> Result<Self, MetalError> {
        if !raw_blocks.len().is_multiple_of(Q8_0_BLOCK_SIZE_BYTES) {
            return Err(MetalError::InvalidOperation(format!(
                "Q8_0 raw length {} is not divisible by block size {}",
                raw_blocks.len(),
                Q8_0_BLOCK_SIZE_BYTES
            )));
        }

        let logical_elems = logical_dims.iter().product::<usize>();
        let blocks = raw_blocks.len() / Q8_0_BLOCK_SIZE_BYTES;
        let total_weights = blocks * Q8_0_WEIGHTS_PER_BLOCK;
        if total_weights < logical_elems {
            return Err(MetalError::InvalidOperation(format!(
                "Q8_0 blocks hold {} weights but logical dims require {}",
                total_weights, logical_elems
            )));
        }

        let mut data_bytes = Vec::with_capacity(blocks * Q8_0_WEIGHTS_PER_BLOCK);
        let mut scale_bytes = Vec::with_capacity(blocks * Q8_0_SCALE_BYTES_PER_BLOCK);
        for chunk in raw_blocks.chunks_exact(Q8_0_BLOCK_SIZE_BYTES) {
            scale_bytes.extend_from_slice(&chunk[0..Q8_0_SCALE_BYTES_PER_BLOCK]);
            data_bytes.extend_from_slice(&chunk[Q8_0_SCALE_BYTES_PER_BLOCK..Q8_0_BLOCK_SIZE_BYTES]);
        }

        let data = upload_u8_bytes(&data_bytes, &ctx.device, &ctx.command_queue)?;
        let scales = upload_u8_bytes(&scale_bytes, &ctx.device, &ctx.command_queue)?;
        let k = logical_dims.first().copied().unwrap_or(0);
        let blocks_per_k = k.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);

        Ok(Self {
            kind: CanonicalQuantKind::Q8_0,
            logical_dims,
            data,
            scales,
            zero_points: None,
            blocks_per_k,
            weights_per_block: Q8_0_WEIGHTS_PER_BLOCK,
        })
    }

    pub fn from_split_q8_tensor(q8: &QuantizedQ8_0Tensor) -> Result<Self, MetalError> {
        // Infer blocks_per_k robustly using buffer sizes and dims.
        let dims = &q8.logical_dims;
        let scales_count = q8.scales.len() / Q8_0_SCALE_BYTES_PER_BLOCK;
        let blocks_per_k = if dims.len() >= 2 {
            let n0 = dims[0];
            let k0 = dims[1];
            let cand0 = k0.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
            let ok0 = cand0.saturating_mul(n0) == scales_count;

            let k1 = dims[0];
            let n1 = dims[1];
            let cand1 = k1.div_ceil(Q8_0_WEIGHTS_PER_BLOCK);
            let ok1 = cand1.saturating_mul(n1) == scales_count;

            if ok0 {
                cand0
            } else if ok1 {
                cand1
            } else {
                k1.div_ceil(Q8_0_WEIGHTS_PER_BLOCK) // Fallback
            }
        } else {
            0
        };

        Ok(Self {
            kind: CanonicalQuantKind::Q8_0,
            logical_dims: q8.logical_dims.clone(),
            data: q8.data.clone(),
            scales: q8.scales.clone(),
            zero_points: None,
            blocks_per_k,
            weights_per_block: Q8_0_WEIGHTS_PER_BLOCK,
        })
    }
}

fn upload_u8_bytes(
    data: &[u8],
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: &Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
) -> Result<Tensor<U8>, MetalError> {
    let byte_len = data.len();
    let dest_buf = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
        .ok_or(MetalError::BufferCreationFailed(byte_len))?;

    let staging_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                core::ptr::NonNull::new(data.as_ptr() as *mut core::ffi::c_void).ok_or(MetalError::NullPointer)?,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(MetalError::BufferFromBytesCreationFailed)?
    };

    let cmd = CommandBuffer::new(command_queue)?;
    let blit = cmd.get_blit_encoder()?;
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging_buf, 0, &dest_buf, 0, byte_len);
    }
    cmd.commit();
    crate::context::command_buffer_pipeline::dispatch_completions(
        command_queue,
        &crate::context::command_buffer_pipeline::wait_with_pipeline(command_queue, &cmd, None),
    );

    Tensor::<U8>::from_existing_buffer(dest_buf, vec![byte_len], Dtype::U8, device, command_queue, 0, false)
}
