use crate::tensor::TensorElement;

/// Identifies the backing allocation strategy for a new [`Tensor`].
///
/// * [`TensorStorage::Dedicated`] allocates a fresh `MTLBuffer` from the
///   provided [`Context`]. This path supports all initialization modes and is
///   the correct choice for long-lived model weights or host-borrowed data.
/// * [`TensorStorage::Pooled`] draws memory from the context's transient bump
///   allocator. It requires a mutable context reference because it advances the
///   pool cursor and should only be used for short-lived activations.
pub enum TensorStorage<'ctx, T: TensorElement> {
    Dedicated(&'ctx super::Context<T>),
    Pooled(&'ctx mut super::Context<T>),
}

/// Describes how the contents of a new [`Tensor`] should be seeded.
///
/// * [`TensorInit::Uninitialized`] leaves the backing buffer untouched.
/// * [`TensorInit::CopyFrom`] copies the provided slice into the allocation.
/// * [`TensorInit::BorrowHost`] wraps an existing host slice without copying
///   and therefore requires dedicated storage.
pub enum TensorInit<'data, T: TensorElement> {
    Uninitialized,
    CopyFrom(&'data [T::Scalar]),
    BorrowHost(&'data [T::Scalar]),
}

impl<'data, T: TensorElement> TensorInit<'data, T> {
    pub fn validate(&self, dims: &[usize]) -> Result<(), crate::MetalError> {
        let expected_elements = dims.iter().product::<usize>();
        match self {
            TensorInit::Uninitialized => Ok(()),
            TensorInit::CopyFrom(data) | TensorInit::BorrowHost(data) => {
                if data.len() != expected_elements {
                    Err(crate::MetalError::DimensionMismatch {
                        expected: expected_elements,
                        actual: data.len(),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }
}
