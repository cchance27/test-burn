use std::hash::{Hash, Hasher};

use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication};
use serde::{Deserialize, Serialize};

use crate::{caching::CacheableKernel, error::MetalError, tensor::dtypes::Dtype};

/// Key for MPS matrix multiplication operations.
///
/// This key uniquely identifies an MPS matrix multiplication operation
/// based on its dimensions and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGemmKey {
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub result_rows: usize,
    pub result_columns: usize,
    pub interior_columns: usize,
    pub batch_size: usize,
    pub alpha: f32,
    pub beta: f32,
    /// Additional specialization factors
    pub beta_nonzero: bool, // Group by beta==0 vs !=0 instead of exact value
    pub dtype: Dtype, // Include dtype for more precise caching
}

impl PartialEq for MpsGemmKey {
    fn eq(&self, other: &Self) -> bool {
        self.transpose_left == other.transpose_left
            && self.transpose_right == other.transpose_right
            && self.result_rows == other.result_rows
            && self.result_columns == other.result_columns
            && self.interior_columns == other.interior_columns
            && self.batch_size == other.batch_size
            && self.alpha == other.alpha
            && self.beta == other.beta
            && self.beta_nonzero == other.beta_nonzero
            && self.dtype == other.dtype
    }
}

impl Eq for MpsGemmKey {}

impl Hash for MpsGemmKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.transpose_left.hash(state);
        self.transpose_right.hash(state);
        self.result_rows.hash(state);
        self.result_columns.hash(state);
        self.interior_columns.hash(state);
        self.batch_size.hash(state);
        self.alpha.to_bits().hash(state);
        self.beta.to_bits().hash(state);
        self.beta_nonzero.hash(state);
        self.dtype.hash(state);
    }
}

/// Key for MPS matrix descriptors.
///
/// This key uniquely identifies an MPS matrix descriptor based on its dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsMatrixDescriptorKey {
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub matrices: usize,
    pub matrix_bytes: usize,
    pub dtype: Dtype,
}

impl PartialEq for MpsMatrixDescriptorKey {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.row_bytes == other.row_bytes
            && self.matrices == other.matrices
            && self.matrix_bytes == other.matrix_bytes
            && self.dtype == other.dtype
    }
}

impl Eq for MpsMatrixDescriptorKey {}

impl Hash for MpsMatrixDescriptorKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.columns.hash(state);
        self.row_bytes.hash(state);
        self.matrices.hash(state);
        self.matrix_bytes.hash(state);
        self.dtype.hash(state);
    }
}

/// Cached MPS GEMM executable plus key metadata.
#[derive(Clone)]
pub struct CacheableMpsGemm {
    pub gemm: Retained<MPSMatrixMultiplication>,
    pub key: MpsGemmKey,
}

impl CacheableMpsGemm {
    pub fn key(&self) -> &MpsGemmKey {
        &self.key
    }

    pub fn from_key(key: &MpsGemmKey, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
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

impl CacheableMpsMatrixDescriptor {
    pub fn key(&self) -> &MpsMatrixDescriptorKey {
        &self.key
    }

    pub fn from_key(key: &MpsMatrixDescriptorKey, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
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
