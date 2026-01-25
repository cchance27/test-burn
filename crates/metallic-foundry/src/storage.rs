use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
use crate::error::MetalError;

use super::Foundry;

/// Marker trait for tensor storage states.
/// This enforces the Typestate pattern for Tensor lifecycle management.
pub trait StorageState: 'static + Clone + std::fmt::Debug {
    fn check_validity(&self) -> Result<(), MetalError> {
        Ok(())
    }

    fn as_view_guard(&self) -> Option<(u64, Arc<AtomicU64>)> {
        None
    }
}

/// State indicating the tensor owns its own dedicated Metal buffer.
/// Safe for long-lived storage.
#[derive(Clone, Copy, Debug)]
pub struct Dedicated;

impl StorageState for Dedicated {}

/// State indicating the tensor borrows memory from the Foundry's ephemeral pool.
/// These tensors are tied to the pool's lifecycle and cannot outlive a reset.
#[derive(Clone, Debug)]
pub struct Pooled {
    pub(crate) generation: u64,
    pub(crate) pool_generation: Arc<AtomicU64>,
}

impl StorageState for Pooled {
    fn check_validity(&self) -> Result<(), MetalError> {
        if self.generation != self.pool_generation.load(Ordering::Relaxed) {
            return Err(MetalError::UseAfterFree);
        }
        Ok(())
    }

    fn as_view_guard(&self) -> Option<(u64, Arc<AtomicU64>)> {
        Some((self.generation, self.pool_generation.clone()))
    }
}

/// State indicating the tensor is a view into another tensor's memory.
#[derive(Clone, Debug)]
pub struct View {
    pub(crate) guard: Option<(u64, Arc<AtomicU64>)>,
}

impl StorageState for View {
    fn check_validity(&self) -> Result<(), MetalError> {
        if let Some((generation, ref pool_gen)) = self.guard {
            if generation != pool_gen.load(Ordering::Relaxed) {
                return Err(MetalError::UseAfterFree);
            }
        }
        Ok(())
    }

    fn as_view_guard(&self) -> Option<(u64, Arc<AtomicU64>)> {
        self.guard.clone()
    }
}

/// Enum describing how to allocate a new Tensor.
/// Used during construction to select the strategy.
pub enum TensorStorage<'a> {
    Dedicated(&'a mut Foundry),
    Pooled(&'a mut Foundry),
}
