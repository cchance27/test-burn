//! Centralized memory layout and stride logic for Foundry kernels.
//!
//! This module provides utilities for calculating strides, padding, and offsets
//! that are consistent across different kernel types (GEMV, GEMM, SDPA).

use serde::{Deserialize, Serialize};

use crate::tensor::Dtype;

/// Memory layout for tensors.
///
/// Unifies layout definitions from across the codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Weights stored as [N, K] (output-major).
    /// Index: weights[n * K + k]
    RowMajor,

    /// Weights stored as [K, N] (input-major).
    /// Index: weights[k * N + n]
    ColMajor,

    /// Canonical Blocked format: [N, K] with K dimension blocked.
    /// Index: weights[(k % wpb) + wpb * (n + (k / wpb) * N)]
    Canonical { expected_k: usize, expected_n: usize },
}

impl Layout {
    pub fn short_name(&self) -> &'static str {
        match self {
            Layout::RowMajor => "rowmajor",
            Layout::ColMajor => "colmajor",
            Layout::Canonical { .. } => "canonical",
        }
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self::RowMajor
    }
}

/// Strategy for tile-aware layout calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutStrategy {
    /// Standard contiguous layout.
    Contiguous,
    /// Head-major layout used in attention: [Heads, Batch, Seq, Dim] or [Heads, Seq, Dim].
    HeadMajor,
    /// Interleaved/Token-major layout: [Batch, Seq, Heads, Dim].
    Interleaved,
}

/// Helper for calculating strides and padding for tiled kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TiledLayout {
    pub batch: u32,
    pub heads: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,

    /// Stride between heads (in elements).
    pub head_stride: u64,
    /// Stride between rows (M dimension, in elements).
    pub row_stride: u64,
    /// Stride between batches (if applicable).
    pub batch_stride: u64,

    /// Total elements including padding.
    pub total_elems: u64,

    /// The tile size used for M-padding.
    pub tile_m: u32,
    /// The actual M dimension after padding.
    pub padded_m: u32,
}

impl TiledLayout {
    /// Create a new layout for SDPA scratch buffers (Scores/Probs).
    /// These are Head-Major: [Heads, M, SeqLen]
    pub fn sdpa_scratch(heads: u32, m: u32, seq_len: u32, tile_m: u32) -> Self {
        // Only pad M if it's > 1 (prefill) to keep decode path lean.
        // GEMM epilogue uses safe stores to handle edge tiles correctly.
        let padded_m = if m > 1 { ((m + tile_m - 1) / tile_m) * tile_m } else { m };

        let row_stride = seq_len as u64;
        let head_stride = (padded_m as u64) * row_stride;
        let total_elems = (heads as u64) * head_stride;

        Self {
            batch: 1,
            heads,
            m,
            n: seq_len,
            k: 0,
            head_stride,
            row_stride,
            batch_stride: total_elems,
            total_elems,
            tile_m,
            padded_m,
        }
    }

    /// Create a layout for token-major output: [M, Heads * HeadDim]
    pub fn interleaved_output(m: u32, heads: u32, head_dim: u32) -> Self {
        let row_stride = (heads as u64) * (head_dim as u64);
        let total_elems = (m as u64) * row_stride;

        Self {
            batch: 1,
            heads,
            m,
            n: head_dim,
            k: 0,
            head_stride: head_dim as u64, // Per-head offset for interleaved output
            row_stride,
            batch_stride: total_elems,
            total_elems,
            tile_m: 1,
            padded_m: m,
        }
    }

    /// Calculate total bytes for a given dtype.
    pub fn total_bytes(&self, dtype: Dtype) -> usize {
        (self.total_elems as usize) * dtype.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdpa_scratch_padding() {
        // Tile size 32
        let layout = TiledLayout::sdpa_scratch(8, 10, 128, 32);
        // M=10 -> padded_m=32
        assert_eq!(layout.padded_m, 32);
        assert_eq!(layout.row_stride, 128);
        assert_eq!(layout.head_stride, 32 * 128);

        // M=1 -> padded_m=1 (no padding for decode)
        let decode = TiledLayout::sdpa_scratch(8, 1, 128, 32);
        assert_eq!(decode.padded_m, 1);
        assert_eq!(decode.head_stride, 128);
    }
}
