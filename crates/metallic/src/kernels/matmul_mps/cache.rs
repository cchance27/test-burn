use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};

use crate::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey}, cacheable::Cacheable, caching::CacheableKernel, error::MetalError
};

/// Cached MPS GEMM executable plus key metadata.
#[derive(Clone)]
pub struct CacheableMpsGemm {
    pub gemm: Retained<MPSMatrixMultiplication>,
    pub key: MpsGemmKey,
}

impl Cacheable for CacheableMpsGemm {
    type Key = MpsGemmKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let device = device.ok_or(MetalError::DeviceNotFound)?;
        let gemm = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                device,
                key.transpose_left,
                key.transpose_right,
                key.result_rows,
                key.result_columns,
                key.interior_columns,
                key.alpha as f64,
                key.beta as f64,
            )
        };
        Ok(Self { gemm, key: key.clone() })
    }
}

/// Cached MPS matrix descriptor plus the original key.
#[derive(Clone)]
pub struct CacheableMpsMatrixDescriptor {
    pub descriptor: Retained<MPSMatrixDescriptor>,
    pub key: MpsMatrixDescriptorKey,
}

impl Cacheable for CacheableMpsMatrixDescriptor {
    type Key = MpsMatrixDescriptorKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let data_type = key.dtype.into();
        let descriptor = unsafe {
            if key.matrices > 1 || key.matrix_bytes != key.row_bytes * key.rows {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    key.rows,
                    key.columns,
                    key.matrices,
                    key.row_bytes,
                    key.matrix_bytes,
                    data_type,
                )
            } else {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(key.rows, key.columns, key.row_bytes, data_type)
            }
        };
        Ok(Self {
            descriptor,
            key: key.clone(),
        })
    }
}

/// Zero-sized type that implements `CacheableKernel` for the batched GEMM pipeline.
pub struct MpsGemmKernel;

impl CacheableKernel for MpsGemmKernel {
    type Key = MpsGemmKey;
    type CachedResource = CacheableMpsGemm;
    type Params = MpsGemmKey;

    const CACHE_NAME: &'static str = "gemm";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsGemm::from_key(key, device)
    }
}

/// Zero-sized type that implements `CacheableKernel` for matrix descriptor caching.
pub struct MpsMatrixDescriptorKernel;

impl CacheableKernel for MpsMatrixDescriptorKernel {
    type Key = MpsMatrixDescriptorKey;
    type CachedResource = CacheableMpsMatrixDescriptor;
    type Params = MpsMatrixDescriptorKey;

    const CACHE_NAME: &'static str = "descriptor";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsMatrixDescriptor::from_key(key, device)
    }
}

/// Helper conversions so callers can work with raw MPS types without re-allocating wrappers.
impl MpsGemmKernel {
    #[inline]
    pub fn extract_gemm(resource: &CacheableMpsGemm) -> Retained<MPSMatrixMultiplication> {
        resource.gemm.clone()
    }
}

impl MpsMatrixDescriptorKernel {
    #[inline]
    pub fn extract_descriptor(resource: &CacheableMpsMatrixDescriptor) -> Retained<MPSMatrixDescriptor> {
        resource.descriptor.clone()
    }
}
