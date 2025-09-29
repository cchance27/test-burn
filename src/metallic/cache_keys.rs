use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::metallic::tensor::dtypes::Dtype;

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
}

impl PartialEq for MpsSoftMaxKey {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns
    }
}

impl Eq for MpsSoftMaxKey {}

impl Hash for MpsSoftMaxKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.columns.hash(state);
    }
}

/// Key for SDPA operations.
///
/// This key uniquely identifies an SDPA operation based on its dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaKey {
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
}

impl PartialEq for SdpaKey {
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch && self.seq_q == other.seq_q && self.seq_k == other.seq_k && self.dim == other.dim
    }
}

impl Eq for SdpaKey {}

impl Hash for SdpaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch.hash(state);
        self.seq_q.hash(state);
        self.seq_k.hash(state);
        self.dim.hash(state);
    }
}
