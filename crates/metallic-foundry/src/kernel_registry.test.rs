#![cfg(test)]

use super::*;
use crate::{Includes, Kernel, KernelSource, tensor::Dtype, types::ComputeCommandEncoder};

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

#[derive(Clone, Copy)]
struct MockRuntimeTypedKernel {
    dtype: Dtype,
    max_linear_index: u64,
}

impl Kernel for MockRuntimeTypedKernel {
    type Args = ();

    fn dtype(&self) -> Option<Dtype> {
        Some(self.dtype)
    }

    fn function_name(&self) -> &str {
        "mock_runtime_typed_kernel"
    }

    fn source(&self) -> KernelSource {
        KernelSource::String("kernel void mock_runtime_typed_kernel() {}".to_string())
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn bind(&self, _encoder: &ComputeCommandEncoder) {}

    fn source_hash(&self) -> u64 {
        42
    }

    fn runtime_dtype_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.dtype.hash(&mut hasher);
        self.max_linear_index.hash(&mut hasher);
        hasher.finish()
    }
}

#[test]
fn test_runtime_dtype_hash_changes_with_runtime_dtype() {
    let f16 = MockRuntimeTypedKernel {
        dtype: Dtype::F16,
        max_linear_index: 1024,
    };
    let f32 = MockRuntimeTypedKernel {
        dtype: Dtype::F32,
        max_linear_index: 1024,
    };

    let f16_hash = f16.runtime_dtype_hash();
    let f32_hash = f32.runtime_dtype_hash();

    assert_ne!(f16_hash, f32_hash);
}

#[test]
fn test_runtime_dtype_hash_changes_with_index_footprint() {
    let small = MockRuntimeTypedKernel {
        dtype: Dtype::F16,
        max_linear_index: 1024,
    };
    let large = MockRuntimeTypedKernel {
        dtype: Dtype::F16,
        max_linear_index: u64::from(u32::MAX) + 1,
    };

    assert_ne!(small.runtime_dtype_hash(), large.runtime_dtype_hash());
}
