use super::{cache_keys::SdpaKey, cacheable::Cacheable, error::MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};

/// A cacheable SDPA (Scaled Dot Product Attention) operation.
///
/// This struct represents a complete SDPA operation that can be cached
/// and reused based on its dimensions.
#[derive(Clone)]
pub struct CacheableSdpa {
    pub key: SdpaKey,
    pub scale: f32,
    // We don't store the actual Metal resources here since they can't be serialized
    // Instead, we store the information needed to recreate them
}

impl Cacheable for CacheableSdpa {
    type Key = SdpaKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(
        key: &Self::Key,
        _device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Self, MetalError> {
        let scale = 1.0 / (key.dim as f32).sqrt();
        Ok(Self {
            key: key.clone(),
            scale,
        })
    }
}
