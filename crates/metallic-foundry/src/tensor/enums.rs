use crate::tensor::TensorElement;

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
