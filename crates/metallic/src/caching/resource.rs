use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication, MPSMatrixSoftMax};

use crate::{
    caching::{CacheMetrics, CacheRegistry, CacheableKernel}, error::MetalError, kernels::{
        kv_cache_write::cache::{CacheableMpsGraphKvWrite, KvWriteGraphKernel, MpsGraphKvWriteKey}, matmul_mps::cache::{MpsGemmKernel, MpsGemmKey, MpsMatrixDescriptorKernel, MpsMatrixDescriptorKey}, scaled_dot_product_attention::cache::{CacheableSdpa, SdpaKernel, SdpaKey}, sdpa_mps_graph::cache::{
            CacheableMpsGraphSdpa, CacheableMpsGraphSdpaMask, MaskSizeBucket, MpsGraphSdpaKernel, MpsGraphSdpaKey, MpsGraphSdpaMaskKernel, MpsGraphSdpaMaskKey
        }, softmax_mps::cache::{MpsSoftMaxKey, SeqKBucket, SoftmaxMpsKernel}
    }, mps_graph::cache::{CacheableMpsGraphFused, FusedOperationType, MpsGraphFusedKernel, MpsGraphFusedKey}, tensor::dtypes::Dtype
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

    /// Retrieve or create a cached MPSGraph SDPA executable.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_sdpa(
        &mut self,
        batch: usize,
        _seq_q: usize,
        _seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<&mut CacheableMpsGraphSdpa, MetalError> {
        let key = MpsGraphSdpaKey {
            batch,
            dim,
            causal,
            dtype,
            accumulator_dtype,
        };
        self.get_or_create_entry::<MpsGraphSdpaKernel>(&key, None)
    }

    /// Retrieve or create the cached SDPA mask tensor arena entry.
    #[inline]
    pub fn get_or_create_mpsgraph_sdpa_mask(
        &mut self,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
    ) -> Result<&mut CacheableMpsGraphSdpaMask, MetalError> {
        let key = MpsGraphSdpaMaskKey {
            causal,
            dtype,
            head_dim: dim,
            seq_q_bucket: MaskSizeBucket::from(seq_q.max(seq_k)),
            seq_k_bucket: MaskSizeBucket::from(seq_k),
        };
        self.get_or_create_entry::<MpsGraphSdpaMaskKernel>(&key, None)
    }

    /// Retrieve both SDPA executable and mask in a single call to minimize borrow churn.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_sdpa_and_mask(
        &mut self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        causal: bool,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<(CacheableMpsGraphSdpa, Option<CacheableMpsGraphSdpaMask>), MetalError> {
        let sdpa_entry = self.get_or_create_mpsgraph_sdpa(batch, seq_q, seq_k, dim, causal, dtype, accumulator_dtype)?;
        let sdpa_clone = sdpa_entry.clone();

        let mask_clone = if causal {
            Some(self.get_or_create_mpsgraph_sdpa_mask(seq_q, seq_k, dim, causal, dtype)?.clone())
        } else {
            None
        };

        Ok((sdpa_clone, mask_clone))
    }

    /// Retrieve or create a cached fused MPSGraph executable.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_mpsgraph_fused(
        &mut self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        output_dim: usize,
        causal: bool,
        operation_type: FusedOperationType,
        dtype: Dtype,
        accumulator_dtype: Option<Dtype>,
    ) -> Result<&mut CacheableMpsGraphFused, MetalError> {
        let key = MpsGraphFusedKey {
            batch,
            seq_q,
            seq_k,
            dim,
            output_dim,
            causal,
            operation_type,
            dtype,
            accumulator_dtype,
        };
        self.get_or_create_entry::<MpsGraphFusedKernel>(&key, None)
    }

    /// Retrieve or create the cached MPSGraph KV write executable.
    #[inline]
    pub fn get_or_create_mpsgraph_kv_write(
        &mut self,
        heads: usize,
        seq_bucket: usize,
        head_dim: usize,
        dtype: Dtype,
    ) -> Result<&mut CacheableMpsGraphKvWrite, MetalError> {
        let key = MpsGraphKvWriteKey {
            heads,
            seq_bucket,
            head_dim,
            dtype,
        };
        self.get_or_create_entry::<KvWriteGraphKernel>(&key, None)
    }

    /// Snapshot metrics for the known cache categories.
    #[inline]
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            gemm: self.metrics::<MpsGemmKernel>(),
            descriptor: self.metrics::<MpsMatrixDescriptorKernel>(),
            softmax: self.metrics::<SoftmaxMpsKernel>(),
            sdpa: self.metrics::<SdpaKernel>(),
            mpsgraph_sdpa: self.metrics::<MpsGraphSdpaKernel>(),
            mpsgraph_mask: self.metrics::<MpsGraphSdpaMaskKernel>(),
            mpsgraph_fused: self.metrics::<MpsGraphFusedKernel>(),
            mpsgraph_kv_write: self.metrics::<KvWriteGraphKernel>(),
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
    pub mpsgraph_sdpa: CacheMetrics,
    pub mpsgraph_mask: CacheMetrics,
    pub mpsgraph_fused: CacheMetrics,
    pub mpsgraph_kv_write: CacheMetrics,
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
        assert_eq!(stats.mpsgraph_sdpa.size, 0);
        assert_eq!(stats.mpsgraph_mask.size, 0);
        assert_eq!(stats.mpsgraph_fused.size, 0);
        assert_eq!(stats.mpsgraph_kv_write.size, 0);
    }
}
