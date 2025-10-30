use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::MPSMatrixSoftMax;

use crate::{cache_keys::MpsSoftMaxKey, cacheable::Cacheable, caching::CacheableKernel, error::MetalError};

/// Cached MPS softmax executable.
#[derive(Clone)]
pub struct CacheableMpsSoftMax {
    pub softmax: Retained<MPSMatrixSoftMax>,
    pub key: MpsSoftMaxKey,
}

impl Cacheable for CacheableMpsSoftMax {
    type Key = MpsSoftMaxKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
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
