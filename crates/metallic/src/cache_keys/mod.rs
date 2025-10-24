pub mod matmul_dispatcher_key;
pub mod mps_graph_kv_write_key;
pub mod softmax_dispatcher_key;

use std::hash::{Hash, Hasher};

pub use matmul_dispatcher_key::*;
pub use mps_graph_kv_write_key::*;
use serde::{Deserialize, Serialize};
pub use softmax_dispatcher_key::*;

use crate::tensor::dtypes::Dtype;

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

/// Key for MPSGraph SDPA operations.
///
/// This key uniquely identifies an MPSGraph SDPA operation based on its dimensions
/// and parameters for proper caching of compiled graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphSdpaKey {
    pub batch: usize,
    pub dim: usize,
    pub causal: bool,
    pub dtype: Dtype,
    pub accumulator_dtype: Option<Dtype>,
}

impl PartialEq for MpsGraphSdpaKey {
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch
            && self.dim == other.dim
            && self.causal == other.causal
            && self.dtype == other.dtype
            && self.accumulator_dtype == other.accumulator_dtype
    }
}

impl Eq for MpsGraphSdpaKey {}

impl Hash for MpsGraphSdpaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch.hash(state);
        self.dim.hash(state);
        self.causal.hash(state);
        self.dtype.hash(state);
        self.accumulator_dtype.hash(state);
    }
}

/// Bucketing for mask sizes to enable reuse across different sequence lengths
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MaskSizeBucket {
    XSmall,   // 1-32
    Small,    // 33-128
    Medium,   // 129-512
    Large,    // 513-1024
    XLarge,   // 1025-2048
    XXLarge,  // 2049-4096
    XXXLarge, // >4096
}

impl From<usize> for MaskSizeBucket {
    fn from(seq_len: usize) -> Self {
        match seq_len {
            0..=32 => MaskSizeBucket::XSmall,
            33..=128 => MaskSizeBucket::Small,
            129..=512 => MaskSizeBucket::Medium,
            513..=1024 => MaskSizeBucket::Large,
            1025..=2048 => MaskSizeBucket::XLarge,
            2049..=4096 => MaskSizeBucket::XXLarge,
            _ => MaskSizeBucket::XXXLarge,
        }
    }
}

/// Key for reusable mask buffers in MPSGraph SDPA.
/// This enables mask reuse across different sequence lengths that fit within the same bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphSdpaMaskKey {
    pub causal: bool,
    pub dtype: Dtype,
    pub head_dim: usize,
    pub seq_q_bucket: MaskSizeBucket,
    pub seq_k_bucket: MaskSizeBucket,
}

impl PartialEq for MpsGraphSdpaMaskKey {
    fn eq(&self, other: &Self) -> bool {
        self.causal == other.causal
            && self.dtype == other.dtype
            && self.head_dim == other.head_dim
            && self.seq_q_bucket == other.seq_q_bucket
            && self.seq_k_bucket == other.seq_k_bucket
    }
}

impl Eq for MpsGraphSdpaMaskKey {}

impl Hash for MpsGraphSdpaMaskKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.causal.hash(state);
        self.dtype.hash(state);
        self.head_dim.hash(state);
        self.seq_q_bucket.hash(state);
        self.seq_k_bucket.hash(state);
    }
}

/// Key for MPSGraph fused operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphFusedKey {
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
    pub output_dim: usize,
    pub causal: bool,
    pub dtype: Dtype,
    pub operation_type: FusedOperationType,
    pub accumulator_dtype: Option<Dtype>,
}

impl PartialEq for MpsGraphFusedKey {
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch
            && self.seq_q == other.seq_q
            && self.seq_k == other.seq_k
            && self.dim == other.dim
            && self.output_dim == other.output_dim
            && self.causal == other.causal
            && self.dtype == other.dtype
            && self.operation_type == other.operation_type
            && self.accumulator_dtype == other.accumulator_dtype
    }
}

impl Eq for MpsGraphFusedKey {}

impl Hash for MpsGraphFusedKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch.hash(state);
        self.seq_q.hash(state);
        self.seq_k.hash(state);
        self.dim.hash(state);
        self.output_dim.hash(state);
        self.causal.hash(state);
        self.dtype.hash(state);
        self.operation_type.hash(state);
        self.accumulator_dtype.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FusedOperationType {
    SdpaProjection,
    // Additional fused operations can be added here
}
