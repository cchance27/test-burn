use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::tensor::dtypes::Dtype;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphKvWriteKey {
    pub heads: usize,
    pub seq_bucket: usize,
    pub head_dim: usize,
    pub dtype: Dtype,
}

impl PartialEq for MpsGraphKvWriteKey {
    fn eq(&self, other: &Self) -> bool {
        self.heads == other.heads && self.seq_bucket == other.seq_bucket && self.head_dim == other.head_dim && self.dtype == other.dtype
    }
}

impl Eq for MpsGraphKvWriteKey {}

impl Hash for MpsGraphKvWriteKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.heads.hash(state);
        self.seq_bucket.hash(state);
        self.head_dim.hash(state);
        self.dtype.hash(state);
    }
}
