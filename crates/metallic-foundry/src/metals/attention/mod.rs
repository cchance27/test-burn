//! Attention building blocks shared across attention kernels.
//!
//! This module intentionally hosts primitives that are not FlashAttention-specific, such as:
//! - KV tiling/load helpers
//! - Attention layout utilities
//!
//! FlashAttention kernels live in `metals::flashattention` and should depend on these primitives,
//! not duplicate them.

pub mod stages;
