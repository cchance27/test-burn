use std::hash::{Hash, Hasher};

use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::MPSMatrixSoftMax;
use serde::{Deserialize, Serialize};

use crate::{caching::CacheableKernel, error::MetalError, tensor::dtypes::Dtype};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SeqKBucket {
    Small,
    Medium,
    Large,
    Other,
}

impl From<usize> for SeqKBucket {
    fn from(seq_len: usize) -> Self {
        match seq_len {
            0..=1024 => SeqKBucket::Small,
            1025..=4096 => SeqKBucket::Medium,
            _ => SeqKBucket::Large,
        }
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

/// Cached MPS softmax executable.
#[derive(Clone)]
pub struct CacheableMpsSoftMax {
    pub softmax: Retained<MPSMatrixSoftMax>,
    pub key: MpsSoftMaxKey,
}

impl CacheableMpsSoftMax {
    pub fn key(&self) -> &MpsSoftMaxKey {
        &self.key
    }

    pub fn from_key(key: &MpsSoftMaxKey, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let device = device.ok_or(MetalError::DeviceNotFound)?;
        let softmax = unsafe { MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), device) };
        Ok(Self { softmax, key: key.clone() })
    }
}

/// Cache adapter for MPS softmax executables.
pub struct SoftmaxMpsKernel;

impl CacheableKernel for SoftmaxMpsKernel {
    type Key = MpsSoftMaxKey;
    type CachedResource = CacheableMpsSoftMax;
    type Params = MpsSoftMaxKey;

    const CACHE_NAME: &'static str = "softmax";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsSoftMax::from_key(key, device)
    }
}

impl SoftmaxMpsKernel {
    #[inline]
    pub fn extract_softmax(resource: &CacheableMpsSoftMax) -> Retained<MPSMatrixSoftMax> {
        resource.softmax.clone()
    }
}
