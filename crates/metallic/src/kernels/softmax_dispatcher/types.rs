use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::hash::Hash;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftmaxBackend {
    Auto,
    MPS,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SoftmaxVariant {
    Auto,
    /// One (or few) simdgroup(s) per row when seq_k <= TG_MAX
    Vec,
    /// Multiple simdgroups per row (segmented)
    Block,
}

impl Display for SoftmaxVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoftmaxVariant::Auto => write!(f, "auto"),
            SoftmaxVariant::Vec => write!(f, "vec"),
            SoftmaxVariant::Block => write!(f, "block"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxPolicy {
    pub backend: SoftmaxBackend,
    pub variant: SoftmaxVariant,
    pub threadgroup_size: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SoftmaxShape {
    pub seq_k: usize,
}
