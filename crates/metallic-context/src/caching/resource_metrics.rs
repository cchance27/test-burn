use crate::caching::{CacheMetrics, ResourceCache};

#[derive(Clone, Debug)]
pub struct CacheMetricsSnapshot {
    pub gemm: CacheMetrics,
    pub descriptor: CacheMetrics,
    pub softmax: CacheMetrics,
    pub sdpa: CacheMetrics,
}

impl From<&ResourceCache> for CacheMetricsSnapshot {
    fn from(cache: &ResourceCache) -> Self {
        let stats = cache.get_stats();
        Self {
            gemm: stats.gemm,
            descriptor: stats.descriptor,
            softmax: stats.softmax,
            sdpa: stats.sdpa,
        }
    }
}
