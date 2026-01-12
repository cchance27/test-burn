use std::hash::{Hash, Hasher};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
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

/// Lightweight cache entry for SDPA scale computations.
#[derive(Clone)]
pub struct CacheableSdpa {
    pub key: SdpaKey,
    pub scale: f32,
}

impl CacheableSdpa {
    pub fn key(&self) -> &SdpaKey {
        &self.key
    }

    pub fn from_key(key: &SdpaKey, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let dim_f32 = key.dim as f32;
        let mut scale = 1.0 / dim_f32.sqrt();
        if scale.is_infinite() || scale.is_nan() {
            scale = 1.0;
        } else {
            scale = scale.clamp(1e-6, 1e6);
        }
        Ok(Self { key: key.clone(), scale })
    }
}

/// Cache adapter for the scalar SDPA helper used in attention kernels.
pub struct SdpaKernel;

impl CacheableKernel for SdpaKernel {
    type Key = SdpaKey;
    type CachedResource = CacheableSdpa;
    type Params = SdpaKey;

    const CACHE_NAME: &'static str = "sdpa";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableSdpa::from_key(key, None)
    }
}
