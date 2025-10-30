use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;

use crate::{cache_keys::SdpaKey, cacheable::Cacheable, caching::CacheableKernel, error::MetalError};

/// Lightweight cache entry for SDPA scale computations.
#[derive(Clone)]
pub struct CacheableSdpa {
    pub key: SdpaKey,
    pub scale: f32,
}

impl Cacheable for CacheableSdpa {
    type Key = SdpaKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
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
