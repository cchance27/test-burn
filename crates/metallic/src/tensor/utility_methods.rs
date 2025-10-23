use std::ops::Range;

impl<T: crate::tensor::TensorElement> super::Tensor<T> {
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * self.dtype.size_bytes()
    }

    /// Compute a strided matrix view for tensors shaped as `[batch, rows, columns]`.
    ///
    /// The returned [`MpsMatrixBatchView`] describes how to interpret the tensor's
    /// memory layout when binding it to MPS matrix kernels. The function also
    /// supports 2-D tensors by treating them as a batch of size 1.
    pub fn as_mps_matrix_batch_view(&self) -> Result<super::MpsMatrixBatchView, crate::MetalError> {
        if self.dims.len() < 2 {
            return Err(crate::MetalError::InvalidShape(
                "MPS matrix view requires at least 2 dimensions".to_string(),
            ));
        }

        let elem_size = self.dtype.size_bytes();

        let (batch, rows, cols, row_stride_elems, matrix_stride_elems) = match self.dims.len() {
            2 => {
                let rows = self.dims[0];
                let cols = self.dims[1];
                let row_stride = if self.strides.len() == 2 { self.strides[0] } else { cols };
                let matrix_stride = rows * row_stride;
                (1, rows, cols, row_stride, matrix_stride)
            }
            3 => {
                let batch = self.dims[0];
                let rows = self.dims[1];
                let cols = self.dims[2];
                let row_stride = if self.strides.len() == 3 { self.strides[1] } else { cols };
                let matrix_stride = if self.strides.len() == 3 {
                    self.strides[0]
                } else {
                    rows * row_stride
                };
                (batch, rows, cols, row_stride, matrix_stride)
            }
            _ => {
                // Treat higher rank tensors as contiguous [batch, rows, cols] by collapsing
                // the leading dimensions into the batch dimension.
                let cols = *self
                    .dims
                    .last()
                    .ok_or_else(|| crate::MetalError::InvalidShape("Tensor has no column dimension".to_string()))?;
                let rows = self
                    .dims
                    .iter()
                    .rev()
                    .nth(1)
                    .copied()
                    .ok_or_else(|| crate::MetalError::InvalidShape("Tensor has no row dimension".to_string()))?;
                let batch = self.len() / (rows * cols);
                let row_stride = cols;
                let matrix_stride = rows * row_stride;
                (batch, rows, cols, row_stride, matrix_stride)
            }
        };

        let row_bytes = row_stride_elems * elem_size;
        let matrix_bytes = matrix_stride_elems * elem_size;

        if matrix_bytes < rows * row_bytes {
            return Err(crate::MetalError::InvalidShape(
                "Tensor strides are too small for requested matrix view".to_string(),
            ));
        }

        Ok(super::MpsMatrixBatchView {
            batch,
            rows,
            columns: cols,
            row_bytes,
            matrix_bytes,
        })
    }

    /// Compute strides for contiguous tensor layout
    pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; dims.len()];
        if !dims.is_empty() {
            strides[dims.len() - 1] = 1;
            for i in (0..dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }
        strides
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product::<usize>()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn flatten(&self) -> Self {
        self.build_view(vec![self.len()], vec![1], self.offset)
    }

    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Self, crate::MetalError> {
        let expected_elements: usize = new_dims.iter().product();
        let actual_elements = self.len();
        if expected_elements != actual_elements {
            return Err(crate::MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: actual_elements,
            });
        }
        Ok(self.build_view(new_dims.clone(), Self::compute_strides(&new_dims), self.offset))
    }

    pub fn slice(&self, range: Range<usize>) -> Result<Self, crate::MetalError> {
        if self.dims.is_empty() {
            return Err(crate::MetalError::InvalidShape("Cannot slice a scalar tensor".to_string()));
        }

        let mut new_dims = self.dims.clone();
        let mut new_offset = self.offset;

        let start = range.start;
        let end = range.end;

        if start > end || end > self.dims[0] {
            return Err(crate::MetalError::InvalidShape(format!(
                "Invalid slice range {:?} for dimension 0 with size {}",
                range, self.dims[0]
            )));
        }

        // Update dimension for the sliced axis
        new_dims[0] = end - start;

        // Update the byte offset into the buffer
        new_offset += start * self.strides[0] * self.dtype.size_bytes();

        Ok(self.build_view(new_dims.clone(), Self::compute_strides(&new_dims), new_offset))
    }

    /// Create a view into a contiguous range along the last dimension of the tensor.
    pub fn slice_last_dim(&self, range: Range<usize>) -> Result<Self, crate::MetalError> {
        if self.dims.is_empty() {
            return Err(crate::MetalError::InvalidShape(
                "Cannot slice last dimension of a scalar tensor".to_string(),
            ));
        }

        let last_dim_index = self.dims.len() - 1;
        let last_dim = self.dims[last_dim_index];

        if range.start > range.end || range.end > last_dim {
            return Err(crate::MetalError::InvalidShape(format!(
                "Invalid slice range {:?} for last dimension with size {}",
                range, last_dim
            )));
        }

        let mut new_dims = self.dims.clone();
        new_dims[last_dim_index] = range.end - range.start;

        let new_strides = self.strides.clone();
        let elem_size = self.dtype.size_bytes();
        let stride = self.strides[last_dim_index];
        let offset_adjust = range.start * stride * elem_size;

        Ok(self.build_view(new_dims, new_strides, self.offset + offset_adjust))
    }

    /// Convert the tensor contents to a `Vec<f32>` using the element conversion rules.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        T::to_f32_vec(self.as_slice())
    }

    /// Create a tensor initialized from an `f32` slice by converting into the element type.
    pub fn from_f32_slice<'ctx>(dims: Vec<usize>, storage: super::TensorStorage<'ctx, T>, data: &[f32]) -> Result<Self, crate::MetalError> {
        let converted = T::from_f32_slice(data);
        Self::new(dims, storage, super::TensorInit::CopyFrom(converted.as_slice()))
    }
}
