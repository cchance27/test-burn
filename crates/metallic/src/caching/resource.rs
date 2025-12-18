use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication, MPSMatrixSoftMax};

use crate::{
    caching::{CacheMetrics, CacheRegistry, CacheableKernel}, error::MetalError, kernels::{
        matmul_mps::cache::{MpsGemmKernel, MpsGemmKey, MpsMatrixDescriptorKernel, MpsMatrixDescriptorKey}, scaled_dot_product_attention::cache::{CacheableSdpa, SdpaKernel, SdpaKey}, softmax_mps::cache::{MpsSoftMaxKey, SeqKBucket, SoftmaxMpsKernel}
    }, tensor::dtypes::Dtype
};

/// Unified resource cache facade that wraps the generic kernel registry.
#[derive(Default)]
pub struct ResourceCache {
    registry: CacheRegistry,
}

impl ResourceCache {
    /// Create a new cache with a fallback Metal device used when callers do not provide one.
    pub fn with_device(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            registry: CacheRegistry::with_device(device),
        }
    }

    /// Access the underlying registry for generic operations.
    #[inline]
    fn get_or_create_entry<K: CacheableKernel>(
        &mut self,
        params: &K::Params,
        explicit_device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut K::CachedResource, MetalError> {
        self.registry.get_or_create::<K>(params, explicit_device)
    }

    /// Retrieve or create a cached GEMM executable.
    #[inline]
    pub fn get_or_create_gemm(
        &mut self,
        key: MpsGemmKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<MPSMatrixMultiplication>, MetalError> {
        let entry = self.get_or_create_entry::<MpsGemmKernel>(&key, Some(device))?;
        Ok(MpsGemmKernel::extract_gemm(entry))
    }

    /// Retrieve or create a cached matrix descriptor.
    #[inline]
    pub fn get_or_create_descriptor(
        &mut self,
        key: MpsMatrixDescriptorKey,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<MPSMatrixDescriptor>, MetalError> {
        let entry = self.get_or_create_entry::<MpsMatrixDescriptorKernel>(&key, Some(device))?;
        Ok(MpsMatrixDescriptorKernel::extract_descriptor(entry))
    }

    /// Retrieve or create a cached softmax executable using default causal settings.
    #[inline]
    pub fn get_or_create_softmax(
        &mut self,
        rows: usize,
        columns: usize,
        dtype: Dtype,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<MPSMatrixSoftMax>, MetalError> {
        let key = MpsSoftMaxKey {
            rows,
            columns,
            seq_k_bucket: SeqKBucket::from(columns),
            causal: false,
            dtype,
        };
        let entry = self.get_or_create_entry::<SoftmaxMpsKernel>(&key, Some(device))?;
        Ok(SoftmaxMpsKernel::extract_softmax(entry))
    }

    /// Retrieve or create a cached softmax executable with explicit causal control.
    #[inline]
    pub fn get_or_create_softmax_full(
        &mut self,
        rows: usize,
        columns: usize,
        dtype: Dtype,
        causal: bool,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Retained<MPSMatrixSoftMax>, MetalError> {
        let key = MpsSoftMaxKey {
            rows,
            columns,
            seq_k_bucket: SeqKBucket::from(columns),
            causal,
            dtype,
        };
        let entry = self.get_or_create_entry::<SoftmaxMpsKernel>(&key, Some(device))?;
        Ok(SoftmaxMpsKernel::extract_softmax(entry))
    }

    /// Retrieve or create a cached SDPA scaling helper using legacy defaults.
    #[inline]
    pub fn get_or_create_sdpa(&mut self, batch: usize, dim: usize, dtype: Dtype) -> CacheableSdpa {
        let key = SdpaKey {
            batch,
            dim,
            dtype,
            causal: false,
            seq_k_bucket: SeqKBucket::Other,
            transpose_k: false,
        };
        self.get_or_create_entry::<SdpaKernel>(&key, None)
            .expect("SdpaKernel::create_cached_resource should be infallible")
            .clone()
    }

    /// Retrieve or create a cached SDPA scaling helper with full specialization parameters.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_sdpa_full(
        &mut self,
        batch: usize,
        dim: usize,
        dtype: Dtype,
        causal: bool,
        seq_k: usize,
        transpose_k: bool,
    ) -> CacheableSdpa {
        let key = SdpaKey {
            batch,
            dim,
            dtype,
            causal,
            seq_k_bucket: SeqKBucket::from(seq_k),
            transpose_k,
        };
        self.get_or_create_entry::<SdpaKernel>(&key, None)
            .expect("SdpaKernel::create_cached_resource should be infallible")
            .clone()
    }

    /// Snapshot metrics for the known cache categories.
    #[inline]
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm: self.metrics::<MpsGemmKernel>(),
            descriptor: self.metrics::<MpsMatrixDescriptorKernel>(),
            softmax: self.metrics::<SoftmaxMpsKernel>(),
            sdpa: self.metrics::<SdpaKernel>(),
        }
    }

    /// Clear every cache entry in the registry.
    #[inline]
    pub fn clear(&mut self) {
        self.registry.clear();
    }

    #[inline]
    fn metrics<K: CacheableKernel>(&self) -> CacheMetrics {
        self.registry.metrics::<K>().unwrap_or_default()
    }
}

/// Aggregated cache statistics grouped by kernel type.
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub gemm: CacheMetrics,
    pub descriptor: CacheMetrics,
    pub softmax: CacheMetrics,
    pub sdpa: CacheMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{caching::CacheableKernel, error::MetalError};

    #[derive(Clone, Debug, PartialEq)]
    struct MockResource(u32);

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct MockKey(u32);

    #[derive(Clone, Debug, PartialEq)]
    struct MockParams(u32);

    struct MockKernel;

    impl CacheableKernel for MockKernel {
        type Key = MockKey;
        type CachedResource = MockResource;
        type Params = MockParams;

        const CACHE_NAME: &'static str = "mock";

        fn create_cache_key(params: &Self::Params) -> Self::Key {
            MockKey(params.0)
        }

        fn create_cached_resource(
            key: &Self::Key,
            _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
        ) -> Result<Self::CachedResource, MetalError> {
            Ok(MockResource(key.0))
        }
    }

    #[test]
    fn registry_tracks_hits_and_misses() {
        let mut registry = CacheRegistry::default();

        let first = registry
            .get_or_create::<MockKernel>(&MockParams(7), None)
            .expect("mock should succeed")
            .clone();
        assert_eq!(first, MockResource(7));

        let second = registry
            .get_or_create::<MockKernel>(&MockParams(7), None)
            .expect("mock should succeed")
            .clone();
        assert_eq!(second, MockResource(7));

        let metrics = registry.metrics::<MockKernel>().expect("metrics available");
        assert_eq!(metrics.size, 1);
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
    }

    #[test]
    fn resource_cache_stats_default_to_zero() {
        let cache = ResourceCache::default();
        let stats = cache.get_stats();

        assert_eq!(stats.gemm.size, 0);
        assert_eq!(stats.descriptor.size, 0);
        assert_eq!(stats.softmax.size, 0);
        assert_eq!(stats.sdpa.size, 0);
    }
}
