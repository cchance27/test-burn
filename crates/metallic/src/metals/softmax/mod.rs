//! Softmax kernels for Foundry.
//!
//! Two high-performance variants:
//! - `SoftmaxVec`: Simdgroup reductions for short-to-medium sequences (256-2047)
//! - `SoftmaxBlock`: Segmented reductions for very long sequences (>4096)
//!
//! Use `Softmax::new()` which automatically selects the best variant based on `seq_k`.

pub mod softmax;
pub mod softmax_block;
pub mod softmax_vec;

pub use softmax::*;
pub use softmax_block::*;
pub use softmax_vec::*;
pub mod step;
