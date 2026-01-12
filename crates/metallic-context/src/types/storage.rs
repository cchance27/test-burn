use std::marker::PhantomData;

use crate::{Tensor, TensorElement};

/// Marker trait for the storage state of a tensor (Pooled, Dedicated, View, etc.).
pub trait StorageState: 'static + Send + Sync {}

/// Represents a tensor backed by the global memory pool.
#[derive(Debug, Clone, Copy)]
pub struct Pooled;
impl StorageState for Pooled {}

/// Represents a tensor backed by a dedicated (owned) buffer.
#[derive(Debug, Clone, Copy)]
pub struct Dedicated;
impl StorageState for Dedicated {}

/// Represents a view into another tensor's storage.
#[derive(Debug, Clone, Copy)]
pub struct View;
impl StorageState for View {}

/// A phantom wrapper to apply typestate to the existing Tensor struct.
#[derive(Clone)]
pub struct TypedTensor<T: TensorElement, S: StorageState = Pooled> {
    pub inner: Tensor<T>,
    _state: PhantomData<S>,
}

impl<T: TensorElement, S: StorageState> TypedTensor<T, S> {
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            inner: tensor,
            _state: PhantomData,
        }
    }
}
