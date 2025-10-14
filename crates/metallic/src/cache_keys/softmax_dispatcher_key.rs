use serde::{Deserialize, Serialize};
use std::hash::Hash;

use crate::tensor::dtypes::Dtype;
use crate::kernels::softmax_dispatcher::types::SoftmaxVariant;

/// Key for softmax dispatcher operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxDispatcherKey {
    pub rows: usize,
    pub columns: usize,
    pub seq_k_bucket: SeqKBucket, // Bounded sequence length for TG sizing
    pub causal: bool, // Causal mask flag
    pub variant: SoftmaxVariant, // Specific softmax variant
    pub dtype: Dtype,
}

impl PartialEq for SoftmaxDispatcherKey {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.seq_k_bucket == other.seq_k_bucket
            && self.causal == other.causal
            && self.variant == other.variant
            && self.dtype == other.dtype
    }
}

impl Eq for SoftmaxDispatcherKey {}

impl std::hash::Hash for SoftmaxDispatcherKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.columns.hash(state);
        self.seq_k_bucket.hash(state);
        self.causal.hash(state);
        self.variant.hash(state);
        self.dtype.hash(state);
    }
}

/// Bucketing for sequence lengths for softmax specialization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SeqKBucket {
    Small,    // <= 1024 (vector softmax)
    Medium,   // 1025-4096 (block softmax)
    Large,    // > 4096 (block softmax with segmentation)
    Other,
}

impl From<usize> for SeqKBucket {
    fn from(seq_k: usize) -> Self {
        match seq_k {
            0..=1024 => SeqKBucket::Small,
            1025..=4096 => SeqKBucket::Medium,
            _ => SeqKBucket::Large,  // Fixed unreachable pattern - covers remaining values
        }
    }
}