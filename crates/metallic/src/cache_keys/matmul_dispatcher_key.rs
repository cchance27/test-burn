use serde::{Deserialize, Serialize};

use crate::{
    kernels::matmul_dispatcher::types::{MatmulVariant, SmallNBucket}, tensor::dtypes::Dtype
};

/// Key for matmul dispatcher operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulDispatcherKey {
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub result_rows: usize,      // m
    pub result_columns: usize,   // n
    pub interior_columns: usize, // k
    pub batch_size: usize,
    pub alpha: f32,
    pub beta: f32,
    /// Additional specialization factors for dispatcher
    pub beta_nonzero: bool, // Group by beta==0 vs !=0 instead of exact value
    pub n_bucket: SmallNBucket, // Small-N specialization
    pub variant: MatmulVariant, // Specific variant for dispatch
    pub dtype: Dtype,
}

impl PartialEq for MatmulDispatcherKey {
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
            && self.n_bucket == other.n_bucket
            && self.variant == other.variant
            && self.dtype == other.dtype
    }
}

impl Eq for MatmulDispatcherKey {}

impl std::hash::Hash for MatmulDispatcherKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
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
        self.n_bucket.hash(state);
        self.variant.hash(state);
        self.dtype.hash(state);
    }
}
