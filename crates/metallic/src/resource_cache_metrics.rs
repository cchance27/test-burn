use crate::{caching::CacheMetrics, resource_cache::ResourceCache};

#[derive(Clone, Debug)]
pub struct CacheMetricsSnapshot {
    pub gemm: CacheMetrics,
    pub descriptor: CacheMetrics,
    pub softmax: CacheMetrics,
    pub sdpa: CacheMetrics,
    pub mpsgraph_sdpa: CacheMetrics,
    pub mpsgraph_mask: CacheMetrics,
    pub mpsgraph_fused: CacheMetrics,
}

impl From<&ResourceCache> for CacheMetricsSnapshot {
    fn from(cache: &ResourceCache) -> Self {
        let stats = cache.get_stats();
        Self {
            gemm: stats.gemm,
            descriptor: stats.descriptor,
            softmax: stats.softmax,
            sdpa: stats.sdpa,
            mpsgraph_sdpa: stats.mpsgraph_sdpa,
            mpsgraph_mask: stats.mpsgraph_mask,
            mpsgraph_fused: stats.mpsgraph_fused,
        }
    }
}
