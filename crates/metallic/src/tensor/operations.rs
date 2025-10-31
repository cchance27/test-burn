use objc2_metal::MTLBlitCommandEncoder as _;

use crate::kernels::{elemwise_add::ElemwiseAddOp, elemwise_div::ElemwiseDivOp, elemwise_mul::ElemwiseMulOp, elemwise_sub::ElemwiseSubOp};

impl<T: crate::tensor::TensorElement> super::Tensor<T> {
    /// Ensure the tensor exposes a contiguous batch view suitable for batched MPS kernels.
    ///
    /// When the tensor represents a strided view into a larger cache (e.g. KV cache history)
    /// the first matrix in each batch may begin `matrix_bytes` bytes apart even if only a
    /// subset of the logical rows are active. Batched MPS operations require each matrix to
    /// be tightly packed, so this helper materializes a compact copy when padding is present.
    pub fn ensure_mps_contiguous_batch(&self, ctx: &mut super::Context<T>) -> Result<(Self, super::MpsMatrixBatchView), crate::MetalError> {
        let view = self.as_mps_matrix_batch_view()?;

        let needs_copy = view.batch > 1 && view.matrix_bytes != view.rows * view.row_bytes;
        if !needs_copy {
            return Ok((self.clone(), view));
        }

        let compact = Self::new(
            self.dims.clone(),
            super::TensorStorage::Pooled(ctx),
            super::TensorInit::Uninitialized,
        )?;

        ctx.prepare_tensors_for_active_cmd(&[self])?;

        let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
        let encoder = command_buffer.get_blit_encoder()?;

        let copy_bytes = view.rows * view.row_bytes;
        for batch_idx in 0..view.batch {
            let src_offset = self.offset + batch_idx * view.matrix_bytes;
            let dst_offset = compact.offset + batch_idx * copy_bytes;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &self.buf,
                    src_offset,
                    &compact.buf,
                    dst_offset,
                    copy_bytes,
                );
            }
        }

        ctx.mark_tensor_pending(&compact);
        ctx.finalize_active_command_buffer_if_latency();

        let compact_view = compact.as_mps_matrix_batch_view()?;
        Ok((compact, compact_view))
    }

    #[inline]
    pub fn permute(&self, permute: &[usize], ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if permute.len() != self.dims.len() {
            return Err(crate::MetalError::InvalidShape(
                "Permutation length must match tensor rank".to_string(),
            ));
        }

        let permute_u32: Vec<u32> = permute.iter().map(|&x| x as u32).collect();

        use crate::kernels::permute::PermuteOp;
        ctx.call::<PermuteOp>((self.clone(), permute_u32))
    }

    /// Element-wise add, returns a new tensor on the same device.
    #[inline]
    pub fn add_elem(&self, other: &Self, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if self.dims != other.dims {
            return Err(crate::MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseAddOp>((self.clone(), other.clone()))
    }

    /// Element-wise sub, returns a new tensor on the same device.
    #[inline]
    pub fn sub_elem(&self, other: &Self, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if self.dims != other.dims {
            return Err(crate::MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseSubOp>((self.clone(), other.clone()))
    }

    /// Element-wise mul, returns a new tensor on the same device.
    #[inline]
    pub fn mul_elem(&self, other: &Self, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if self.dims != other.dims {
            return Err(crate::MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseMulOp>((self.clone(), other.clone()))
    }

    #[inline]
    pub fn div_elem(&self, other: &Self, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        if self.dims != other.dims {
            return Err(crate::MetalError::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        ctx.call::<ElemwiseDivOp>((self.clone(), other.clone()))
    }

    /// Element-wise scalar add.
    pub fn add_scalar(&self, value: f32, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        // DEBT: This is inefficient. A dedicated kernel for scalar operations would be better.
        let mut scalar_tensor = Self::zeros_like(self, ctx)?;
        scalar_tensor.fill(value);
        self.add_elem(&scalar_tensor, ctx)
    }

    pub fn get_batch(&self, batch_index: usize) -> Result<Self, crate::MetalError> {
        if self.dims.len() < 3 {
            return Err(crate::MetalError::InvalidShape(
                "get_batch requires at least 3 dimensions".to_string(),
            ));
        }
        if batch_index >= self.dims[0] {
            return Err(crate::MetalError::InvalidShape("batch_index out of bounds".to_string()));
        }

        let elem_size = self.dtype.size_bytes();
        let batch_stride_elems = if self.strides.len() == self.dims.len() && !self.strides.is_empty() {
            self.strides[0]
        } else {
            self.dims[1..].iter().product::<usize>()
        };
        let new_offset = self.offset + batch_index * batch_stride_elems * elem_size;
        let new_strides = if self.strides.len() >= 2 {
            self.strides[1..].to_vec()
        } else {
            Self::compute_strides(&self.dims[1..])
        };

        Ok(self.build_view(self.dims[1..].to_vec(), new_strides, new_offset))
    }

    /// Check tensor values for numerical stability issues
    pub fn validate_numerical_stability(&self) -> Result<(), crate::MetalError> {
        let data = self.as_slice();
        for (i, &val) in data.iter().enumerate() {
            if !T::is_finite(val) {
                return Err(crate::MetalError::InvalidOperation(format!(
                    "Non-finite value detected at index {}: {} in tensor with shape {:?}",
                    i,
                    T::to_f32(val),
                    self.dims
                )));
            }
            // Check for extremely large values that might cause overflow in subsequent operations
            if T::to_f32(T::abs(val)) > 1e6 {
                eprintln!(
                    "Warning: Very large value detected at index {}: {} in tensor with shape {:?}. This could cause numerical instability.",
                    i,
                    T::to_f32(val),
                    self.dims
                );
            }
        }
        Ok(())
    }

    /// Element-wise scalar multiplication
    pub fn mul_scalar(&self, scalar: f32, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        let mut scalar_tensor = Self::zeros_like(self, ctx)?;
        scalar_tensor.fill(scalar);
        self.mul_elem(&scalar_tensor, ctx)
    }

    /// Element-wise absolute value
    pub fn abs(&self, ctx: &mut super::Context<T>) -> Result<Self, crate::MetalError> {
        use crate::kernels::elemwise_abs::ElemwiseAbsOp;
        ctx.call::<ElemwiseAbsOp>((self.clone(),))
    }

    /// Find the maximum scalar value in the tensor
    pub fn max_scalar(&self) -> f32 {
        let data = self.as_slice();
        data.iter().map(|&val| T::to_f32(val)).fold(f32::NEG_INFINITY, |a, b| a.max(b))
    }
}
