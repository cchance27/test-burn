#![cfg(test)]

use super::*;

#[test]
fn test_kernel_cache_key_equality() {
    let key1 = KernelCacheKey::new("gemm", "f16_f16_nn");
    let key2 = KernelCacheKey::new("gemm", "f16_f16_nn");
    let key3 = KernelCacheKey::new("gemm", "q8_f16_nn");

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn test_registry_config_default() {
    let config = RegistryConfig::default();
    assert_eq!(config.max_kernels, 256);
    assert_eq!(config.max_pipelines, 512);
    assert_eq!(config.kernel_ttl, Duration::from_secs(600));
}

#[test]
fn test_cache_stats() {
    let registry = KernelRegistry::new(RegistryConfig::default());
    let stats = registry.stats();
    assert_eq!(stats.kernel_count, 0);
    assert_eq!(stats.pipeline_count, 0);
}
