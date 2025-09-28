use super::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey, MpsSoftMaxKey},
    cacheable::Cacheable,
    error::MetalError,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication, MPSMatrixSoftMax};

/// A cacheable wrapper for MPSMatrixMultiplication.
///
/// This struct implements the Cacheable trait, allowing MPS matrix multiplication
/// operations to be cached and reused based on their parameters.
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

/// A cacheable wrapper for MPSMatrixDescriptor.
///
/// This struct implements the Cacheable trait, allowing MPS matrix descriptors
/// to be cached and reused based on their dimensions.
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
        let descriptor = unsafe {
            if key.matrices > 1 || key.matrix_bytes != key.row_bytes * key.rows {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    key.rows,
                    key.columns,
                    key.matrices,
                    key.row_bytes,
                    key.matrix_bytes,
                    objc2_metal_performance_shaders::MPSDataType::Float32,
                )
            } else {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    key.rows,
                    key.columns,
                    key.row_bytes,
                    objc2_metal_performance_shaders::MPSDataType::Float32,
                )
            }
        };
        Ok(Self {
            descriptor,
            key: key.clone(),
        })
    }
}

/// A cacheable wrapper for `MPSMatrixSoftMax` operations.
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
