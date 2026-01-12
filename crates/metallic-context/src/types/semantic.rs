use super::storage::{Dedicated, Pooled, StorageState, TypedTensor};
use crate::TensorElement;

/// Represents the Query tensor in Attention mechanisms.
#[derive(Clone)]
pub struct Query<T: TensorElement, S: StorageState = Pooled>(pub TypedTensor<T, S>);

/// Represents the Key tensor in Attention mechanisms.
#[derive(Clone)]
pub struct Key<T: TensorElement, S: StorageState = Pooled>(pub TypedTensor<T, S>);

/// Represents the Value tensor in Attention mechanisms.
#[derive(Clone)]
pub struct Value<T: TensorElement, S: StorageState = Pooled>(pub TypedTensor<T, S>);

/// Represents Model Weights (usually Dedicated/Static).
#[derive(Clone)]
pub struct Weight<T: TensorElement>(pub TypedTensor<T, Dedicated>);

/// Represents a Volume (e.g. 3D tensor or batch of images).
#[derive(Clone)]
pub struct Volume<T: TensorElement, S: StorageState = Pooled>(pub TypedTensor<T, S>);
