use super::Foundry;

/// Marker trait for tensor storage states.
/// This enforces the Typestate pattern for Tensor lifecycle management.
pub trait StorageState: 'static {}

/// State indicating the tensor owns its own dedicated Metal buffer.
/// Safe for long-lived storage.
pub struct Dedicated;
impl StorageState for Dedicated {}

/// State indicating the tensor borrows memory from the Foundry's ephemeral pool.
/// These tensors are tied to the pool's lifecycle and cannot outlive a reset.
pub struct Pooled;
impl StorageState for Pooled {}

/// State indicating the tensor is a view into another tensor's memory.
pub struct View;
impl StorageState for View {}

/// Enum describing how to allocate a new Tensor.
/// Used during construction to select the strategy.
pub enum TensorStorage<'a> {
    Dedicated(&'a mut Foundry),
    Pooled(&'a mut Foundry),
}
