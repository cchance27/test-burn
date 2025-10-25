# Improved Caching Infrastructure for Metallic

## Overview

This document outlines the proposed improvements to the metallic crate's caching infrastructure. Currently, there are multiple caching systems that lack a unified approach, leading to architectural issues where kernel functionality bleeds into the caching infrastructure. This report analyzes the current systems and proposes a unified trait-based solution.

## Current Caching Infrastructure

### 1. Resource Cache System (core caching)

The primary caching system in metallic consists of:

#### `cacheable::Cacheable` trait
- Defines the contract for cacheable resources
- Has associated `Key` type
- Provides `cache_key()` and `from_key()` methods
- Implemented by various kernel-specific types

#### `resource_cache::ResourceCache` struct
- Contains hardcoded caches for specific kernel types
- Maintains separate hashmaps for different resource types:
  - `gemm_cache: FxHashMap<MpsGemmKey, CacheEntry<CacheableMpsGemm>>`
  - `softmax_cache: FxHashMap<MpsSoftMaxKey, CacheEntry<CacheableMpsSoftMax>>`
  - `sdpa_cache: FxHashMap<SdpaKey, CacheEntry<CacheableSdpa>>`
  - `mpsgraph_sdpa_cache: GraphExecutableCache<MpsGraphSdpaKey, CacheableMpsGraphSdpa>`
  - `mpsgraph_fused_cache: GraphExecutableCache<MpsGraphFusedKey, CacheableMpsGraphFused>`
  - `mpsgraph_kv_write_cache: GraphExecutableCache<MpsGraphKvWriteKey, CacheableMpsGraphKvWrite>`

#### `cacheable_resources` module
- Contains kernel-specific cacheable wrapper structs
- Each implements the `Cacheable` trait
- Contains kernel creation logic in `from_key()` methods

#### `cache_keys` module
- Defines the key types for caching
- Provides unique identification for cached resources

### 2. KV Cache System (context/kv_cache.rs)

A specialized caching system for Key-Value attention caches:

- Manages KV cache allocations for transformer models
- Tracks cache entries per layer
- Handles cache writes using both kernel and blit operations
- Integrates with MPS graph execution for optimized cache operations

### 3. Tensor Preparation Cache System (tensor_preparation_cache.rs)

A performance optimization cache:

- Caches tensor preparation states
- Avoids redundant preparation operations
- Tracks performance metrics for cache effectiveness
- Uses mutex-protected shared state for thread safety

### 4. Cache-related testing infrastructure

- `tests/cacheable_test.rs` - Tests for cacheable functionality
- `tests/resource_cache_persistence_test.rs` - Tests for cache persistence

## Issues with Current Architecture

### 1. Tight Coupling
- The `ResourceCache` knows about specific kernel types
- Cacheable resources contain kernel-specific creation logic
- Kernels must use the cache infrastructure directly

### 2. Hardcoded Dependencies
- Adding new kernel types requires modifying the core cache
- The cache struct has hardcoded fields for each type
- Method proliferation: `get_or_create_gemm()`, `get_or_create_softmax()`, etc.

### 3. Code Duplication
- Similar `from_key` methods exist across different cacheable types
- Each kernel has its own way of handling cache creation

### 4. Limited Extensibility
- Difficult to add new cache strategies
- Custom caching logic per kernel is awkward to implement
- Type-unsafe additions to the core cache

### 5. Mixed Responsibilities
- Cache infrastructure handles kernel-specific creation logic
- Kernel modules depend on cache infrastructure for resource creation
- Separation of concerns is violated

## Proposed Solution: Unified Trait-Based Architecture

### 1. Core Trait Design

```rust
/// Trait for kernels that support caching
pub trait CacheableKernel {
    /// The type of key used to uniquely identify cached resources
    type Key: Clone + Hash + Eq + Debug + Send + Sync;
    
    /// The type of cached resource this kernel manages
    type CachedResource: Clone + Send + Sync;
    
    /// Create a cache key from operation parameters
    fn create_cache_key(params: Self::Params) -> Self::Key;
    
    /// Create the cached resource from its key
    fn create_cached_resource(
        key: &Self::Key, 
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>
    ) -> Result<Self::CachedResource, MetalError>;
    
    /// Parameters needed to create a cache entry
    type Params;
}
```

### 2. Generic ResourceCache

```rust
pub struct ResourceCache {
    // Type-erased storage for different kernel caches
    caches: HashMap<TypeId, Box<dyn AnyCache>>,
    
    // Default device fallback
    default_device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
}

impl ResourceCache {
    /// Generic method to get or create cached resources for any kernel
    pub fn get_or_create<K: CacheableKernel>(
        &mut self,
        params: K::Params,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>
    ) -> Result<K::CachedResource, MetalError> {
        let key = K::create_cache_key(params);
        let cache = self.get_kernel_cache::<K>();
        cache.get_or_create(&key, device)
    }
    
    /// Type-safe access to kernel-specific caches
    fn get_kernel_cache<K: CacheableKernel>(&mut self) -> &mut KernelCache<K> {
        let type_id = TypeId::of::<K>();
        self.caches
            .entry(type_id)
            .or_insert_with(|| Box::new(KernelCache::<K>::new()))
            .downcast_mut::<KernelCache<K>>()
            .expect("Type should match")
    }
}
```

### 3. Kernel Implementation Example

```rust
// In src/kernels/sdpa_mps_graph/
pub struct SdpaMpsGraphKernel;

// Define kernel-specific parameters
pub struct SdpaParams {
    pub batch: usize,
    pub dim: usize,
    pub causal: bool,
    pub dtype: Dtype,
    pub accumulator_dtype: Option<Dtype>,
}

impl CacheableKernel for SdpaMpsGraphKernel {
    type Key = MpsGraphSdpaKey;
    type CachedResource = CacheableMpsGraphSdpa;
    type Params = SdpaParams;
    
    fn create_cache_key(params: Self::Params) -> Self::Key {
        MpsGraphSdpaKey {
            batch: params.batch,
            dim: params.dim,
            causal: params.causal,
            dtype: params.dtype,
            accumulator_dtype: params.accumulator_dtype,
        }
    }
    
    fn create_cached_resource(
        key: &Self::Key, 
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>
    ) -> Result<Self::CachedResource, MetalError> {
        // Move the existing graph creation logic here
        // This ensures the kernel owns its caching logic
        // ... detailed implementation
        Ok(CacheableMpsGraphSdpa { /* ... */ })
    }
}
```

## Benefits of the Unified Approach

### 1. Complete Decoupling
- Kernels own their caching logic completely
- Cache infrastructure is generic and reusable
- No hardcoded dependencies between cache and kernel types

### 2. Enhanced Type Safety
- Strong typing for each kernel's cache operations
- Compile-time verification of correct usage
- Elimination of runtime type errors

### 3. Superior Extensibility
- Easy addition of new kernel types without cache modifications
- Custom caching strategies per kernel
- Clean separation of caching concerns

### 4. Performance Maintenance
- Zero-cost abstractions through Rust traits
- Monomorphization for specialized code paths
- Potential for better compiler optimization

### 5. Improved Developer Experience
- Clearer API for kernel developers
- Intuitive caching interface
- Better testability of individual components

## Migration Strategy

### Phase 1: Core Infrastructure
- Implement the `CacheableKernel` trait
- Create generic `ResourceCache` structure
- Implement type-erased cache storage

### Phase 2: Kernel Migration
- Move each kernel to implement the new trait
- Transfer kernel creation logic from cacheable resources to kernels
- Update kernel modules to be self-contained

### Phase 3: Integration
- Update `Context` and `KernelInvocable` integration
- Replace direct cache calls with trait-based calls
- Maintain backward compatibility during transition

### Phase 4: Advanced Features
- Extend to other caching systems (KV cache, tensor preparation)
- Implement unified metrics and monitoring
- Add cache eviction policies

## Comparison: Before vs After

| Aspect | Current Approach | Proposed Approach |
|--------|------------------|-------------------|
| Coupling | High (cache knows kernel details) | Low (generic cache) |
| Extensibility | Difficult (requires cache changes) | Easy (implement trait) |
| Type Safety | Moderate (runtime checks) | High (compile-time checks) |
| Performance | Good (specialized) | Same or better (zero-cost) |
| Code Duplication | Present | Eliminated |
| Maintainability | Challenging | Improved |
| Testability | Limited | Enhanced |

## Performance Considerations

The proposed approach maintains performance through:
- **Zero-overhead abstraction**: Rust's trait system
- **Monomorphization**: Specialized code per kernel
- **Efficient type erasure**: Minimal overhead in cache layer
- **Caching effectiveness**: Same underlying mechanisms

## Integration with Other Caching Systems

The unified approach can potentially integrate with or provide patterns for:
- KV cache system: Could adopt similar trait pattern
- Tensor preparation cache: Could benefit from unified metrics
- Future caching needs: Built-in extensibility

## Conclusion

The proposed trait-based caching infrastructure addresses the identified architectural issues while maintaining performance and providing significant improvements in maintainability, extensibility, and developer experience. The approach is fully compatible with Rust idioms and provides a solid foundation for the metallic crate's continued development and growth.