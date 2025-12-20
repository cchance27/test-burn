use super::{Dtype, Tensor, TensorElement, TensorInit, TensorStorage};
use crate::{Context, MetalError};

pub const F16_CANONICAL_WEIGHTS_PER_BLOCK: usize = 32;

#[derive(Clone)]
pub struct CanonicalF16Tensor<T: TensorElement> {
    pub data: Tensor<T>,
    pub logical_dims: Vec<usize>,
    pub blocks_per_k: usize,
}

impl<T: TensorElement> CanonicalF16Tensor<T> {
    pub fn new(logical_dims: Vec<usize>, ctx: &mut Context<T>) -> Result<Self, MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "CanonicalF16Tensor::new",
                dtype: T::DTYPE,
            });
        }
        if logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "CanonicalF16Tensor expects 2D logical dims, got {:?}",
                logical_dims
            )));
        }
        let k = logical_dims[0];
        let n = logical_dims[1];
        if k == 0 || n == 0 {
            return Err(MetalError::InvalidShape(format!(
                "CanonicalF16Tensor requires non-zero dims, got {:?}",
                logical_dims
            )));
        }
        let blocks_per_k = k.div_ceil(F16_CANONICAL_WEIGHTS_PER_BLOCK);
        let len = blocks_per_k
            .checked_mul(n)
            .and_then(|v| v.checked_mul(F16_CANONICAL_WEIGHTS_PER_BLOCK))
            .ok_or_else(|| MetalError::InvalidShape("CanonicalF16Tensor length overflow".to_string()))?;
        let data = Tensor::new(vec![len], TensorStorage::Dedicated(ctx), TensorInit::Uninitialized)?;
        Ok(Self {
            data,
            logical_dims,
            blocks_per_k,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn write_from_nk_tensor(&mut self, src: &Tensor<T>, dst_col_offset: usize) -> Result<(), MetalError> {
        self.write_from_nk_slice(src.as_slice(), src.dims(), dst_col_offset)
    }

    /// Swizzle NK row-major weights (rows = N/out, cols = K/in) into canonical k-block-major layout.
    pub fn write_from_nk_slice(&mut self, src: &[T::Scalar], src_dims: &[usize], dst_col_offset: usize) -> Result<(), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "CanonicalF16Tensor::write_from_nk_slice",
                dtype: T::DTYPE,
            });
        }
        if src_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!("Expected 2D source dims, got {:?}", src_dims)));
        }
        let src_len = src_dims.iter().product::<usize>();
        if src_len != src.len() {
            return Err(MetalError::InvalidShape(format!(
                "Source len {} does not match dims {:?}",
                src.len(),
                src_dims
            )));
        }

        let k = self.logical_dims[0];
        let n_total = self.logical_dims[1];

        let (out_features, in_features) = if src_dims[0] == k {
            (src_dims[1], src_dims[0])
        } else if src_dims[1] == k {
            (src_dims[0], src_dims[1])
        } else {
            return Err(MetalError::InvalidShape(format!(
                "Source dims {:?} do not match expected K={}",
                src_dims, k
            )));
        };

        if dst_col_offset + out_features > n_total {
            return Err(MetalError::InvalidShape(format!(
                "Column range [{}..{}) exceeds canonical N {}",
                dst_col_offset,
                dst_col_offset + out_features,
                n_total
            )));
        }

        let weights_per_block = F16_CANONICAL_WEIGHTS_PER_BLOCK;
        let blocks_per_k = self.blocks_per_k;
        let dst = self.data.as_mut_slice();

        for out_idx in 0..out_features {
            let dst_col = dst_col_offset + out_idx;
            let src_base = out_idx * in_features;

            for block in 0..blocks_per_k {
                let k_base = block * weights_per_block;
                let dst_base = (block * n_total + dst_col) * weights_per_block;
                let remaining = k.saturating_sub(k_base);

                if remaining >= weights_per_block {
                    let src_slice = &src[src_base + k_base..src_base + k_base + weights_per_block];
                    let dst_slice = &mut dst[dst_base..dst_base + weights_per_block];
                    dst_slice.copy_from_slice(src_slice);
                } else {
                    if remaining > 0 {
                        let src_slice = &src[src_base + k_base..src_base + k_base + remaining];
                        let dst_slice = &mut dst[dst_base..dst_base + remaining];
                        dst_slice.copy_from_slice(src_slice);
                    }
                    for pad in remaining..weights_per_block {
                        dst[dst_base + pad] = T::from_f32(0.0);
                    }
                }
            }
        }

        Ok(())
    }
}
