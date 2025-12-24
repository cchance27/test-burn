//! Softmax kernels for Foundry.
//!
//! Two high-performance variants:
//! - `SoftmaxVec`: Simdgroup reductions for short-to-medium sequences (256-2047)
//! - `SoftmaxBlock`: Segmented reductions for very long sequences (>4096)

pub mod softmax;
pub mod softmax_block;
pub mod softmax_vec;

pub use softmax::*;
pub use softmax_block::*;
pub use softmax_vec::*;

/// Softmax variant selection based on sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxVariant {
    /// Vectorized simdgroup reductions (short-medium sequences)
    Vec,
    /// Block-based segmented reductions (long sequences)
    Block,
}

/// Selects the appropriate softmax variant based on seq_k.
/// Based on legacy dispatcher.rs heuristics.
pub fn select_softmax_variant(seq_k: usize) -> SoftmaxVariant {
    match seq_k {
        0..=255 => SoftmaxVariant::Vec, // Fallback to vec for short
        256..=767 => SoftmaxVariant::Vec,
        768..=895 => SoftmaxVariant::Block,
        896..=1023 => SoftmaxVariant::Vec,
        1024..=1279 => SoftmaxVariant::Block,
        1280..=2047 => SoftmaxVariant::Vec,
        2048..=4095 => SoftmaxVariant::Vec, // Vec still competitive
        _ => SoftmaxVariant::Block,         // Very long sequences use block
    }
}
