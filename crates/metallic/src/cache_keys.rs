use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::tensor::dtypes::Dtype;

// Re-export dispatcher-specific cache keys
pub mod matmul_dispatcher_key;
pub mod softmax_dispatcher_key;
pub use matmul_dispatcher_key::*;
pub use softmax_dispatcher_key::*;

/// Key for MPS matrix multiplication operations.
///
/// This key uniquely identifies an MPS matrix multiplication operation
/// based on its dimensions and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGemmKey {
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub result_rows: usize,
    pub result_columns: usize,
    pub interior_columns: usize,
    pub batch_size: usize,
    pub alpha: f32,
    pub beta: f32,
    /// Additional specialization factors
    pub beta_nonzero: bool, // Group by beta==0 vs !=0 instead of exact value
    pub dtype: Dtype, // Include dtype for more precise caching
}

impl PartialEq for MpsGemmKey {
    fn eq(&self, other: &Self) -> bool {
        self.transpose_left == other.transpose_left
            && self.transpose_right == other.transpose_right
            && self.result_rows == other.result_rows
            && self.result_columns == other.result_columns
            && self.interior_columns == other.interior_columns
            && self.batch_size == other.batch_size
            && self.alpha == other.alpha
            && self.beta == other.beta
            && self.beta_nonzero == other.beta_nonzero
            && self.dtype == other.dtype
    }
}

impl Eq for MpsGemmKey {}

impl Hash for MpsGemmKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.transpose_left.hash(state);
        self.transpose_right.hash(state);
        self.result_rows.hash(state);
        self.result_columns.hash(state);
        self.interior_columns.hash(state);
        self.batch_size.hash(state);
        // For f32, we need to be careful about NaN values
        // We'll use the bit representation for hashing
        self.alpha.to_bits().hash(state);
        self.beta.to_bits().hash(state);
        self.beta_nonzero.hash(state);
        self.dtype.hash(state);
    }
}

/// Key for MPS matrix descriptors.
///
/// This key uniquely identifies an MPS matrix descriptor based on its dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsMatrixDescriptorKey {
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub matrices: usize,
    pub matrix_bytes: usize,
    pub dtype: Dtype,
}

impl PartialEq for MpsMatrixDescriptorKey {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.row_bytes == other.row_bytes
            && self.matrices == other.matrices
            && self.matrix_bytes == other.matrix_bytes
            && self.dtype == other.dtype
    }
}

impl Eq for MpsMatrixDescriptorKey {}

impl Hash for MpsMatrixDescriptorKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.columns.hash(state);
        self.row_bytes.hash(state);
        self.matrices.hash(state);
        self.matrix_bytes.hash(state);
        self.dtype.hash(state);
    }
}

/// Key for cached MPS softmax operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsSoftMaxKey {
    pub rows: usize,
    pub columns: usize,
    pub seq_k_bucket: SeqKBucket, // Bounded sequence length for TG sizing
    pub causal: bool,             // Causal mask flag
    pub dtype: Dtype,
}

impl PartialEq for MpsSoftMaxKey {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.seq_k_bucket == other.seq_k_bucket
            && self.causal == other.causal
            && self.dtype == other.dtype
    }
}

impl Eq for MpsSoftMaxKey {}

impl Hash for MpsSoftMaxKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.columns.hash(state);
        self.seq_k_bucket.hash(state);
        self.causal.hash(state);
        self.dtype.hash(state);
    }
}

/// Key for SDPA operations.
///
/// This key uniquely identifies an SDPA operation based on attributes that
/// remain stable throughout a decoding session. Sequence lengths are tracked
/// separately so that incremental decoding can continue to hit the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaKey {
    pub batch: usize,
    pub dim: usize,
    pub dtype: Dtype,
    /// Additional specialization factors for SDPA
    pub causal: bool, // Causal mask flag
    pub seq_k_bucket: SeqKBucket, // Sequence length bucket for softmax specialization
    pub transpose_k: bool,        // Logical transpose preference flag
}

impl PartialEq for SdpaKey {
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch
            && self.dim == other.dim
            && self.dtype == other.dtype
            && self.causal == other.causal
            && self.seq_k_bucket == other.seq_k_bucket
            && self.transpose_k == other.transpose_k
    }
}

impl Eq for SdpaKey {}

impl Hash for SdpaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch.hash(state);
        self.dim.hash(state);
        self.dtype.hash(state);
        self.causal.hash(state);
        self.seq_k_bucket.hash(state);
        self.transpose_k.hash(state);
    }
}
