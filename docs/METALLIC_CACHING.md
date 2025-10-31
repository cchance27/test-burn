# Metallic Caching Infrastructure

## Overview

Metallic's caching system provides a unified, type-safe infrastructure for caching GPU resources (compute pipeline states, descriptors, graphs) and intermediate computation results. The system is designed for:

- **Performance**: Zero-cost abstractions with compile-time type safety
- **Flexibility**: Per-cache eviction policies and configuration
- **Observability**: Full instrumentation with JSONL logging
- **Safety**: Strongly-typed errors, defensive programming

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Code                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ ResourceCache  │  (Convenience Facade)
                    │   (Optional)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ CacheRegistry  │  (Type-erased Storage)
                    └───────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼────────┐  ┌──▼──────┐  ┌────▼─────┐
    │ KernelCache<K> │  │  Slots  │  │  Metrics │
    └───────┬────────┘  └─────────┘  └──────────┘
            │
    ┌───────▼────────┐
    │CacheableKernel │  (Trait implemented by kernels)
    └────────────────┘
```

### Three-Layer Design

1. **CacheableKernel Trait** - Contract for cacheable operations
2. **CacheRegistry** - Generic, type-erased storage for all caches
3. **ResourceCache** - Optional convenience facade with typed methods

## Implementing a Cached Kernel

### Step 1: Define Your Cache Key

The cache key must uniquely identify a cached resource configuration:

```rust
use metallic::caching::CacheableKernel;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MyKernelKey {
    size: usize,
    dtype: Dtype,
    transpose: bool,
}
```

**Key Requirements:**
- Must implement: `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`
- Should be cheap to clone (prefer `Copy` where possible)
- Must uniquely identify the resource

**Design Guidelines:**
- Include all parameters that affect resource compilation
- Use bucketing for continuous values (e.g., round sizes to multiples of 256)
- Avoid over-granular keys that fragment the cache

### Step 2: Define Your Cached Resource

```rust
use objc2::rc::Retained;
use objc2_metal::MTLComputePipelineState;

#[derive(Clone)]
struct MyCachedResource {
    pipeline: Retained<MTLComputePipelineState>,
    thread_group_size: MTLSize,
}
```

**Resource Requirements:**
- Must implement `Clone` (can use `Retained<T>` for Metal objects)
- Should contain all precomputed/compiled artifacts
- Keep resources lightweight (large data should be referenced, not embedded)

### Step 3: Implement CacheableKernel

```rust
use metallic::caching::CacheableKernel;
use metallic::error::MetalError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;

struct MyKernel;

impl CacheableKernel for MyKernel {
    type Key = MyKernelKey;
    type CachedResource = MyCachedResource;
    type Params = (usize, Dtype, bool);  // Input parameters

    const CACHE_NAME: &'static str = "my_kernel";

    fn create_cache_key(params: &Self::Params) -> Self::Key {
        MyKernelKey {
            size: params.0,
            dtype: params.1,
            transpose: params.2,
        }
    }

    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        let device = device.ok_or(MetalError::DeviceNotAvailable)?;
        
        // Compile pipeline state (expensive operation)
        let pipeline = compile_pipeline(device, key)?;
        
        Ok(MyCachedResource {
            pipeline,
            thread_group_size: MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        })
    }
}
```

**Implementation Guidelines:**
- `CACHE_NAME`: Must be unique across all kernel types
- `create_cache_key()`: Should be fast (no allocations if possible)
- `create_cached_resource()`: Can be expensive (only called on cache miss)
- Return proper errors (avoid `unwrap()` or `expect()`)

### Step 4: Use the Cache

#### Via Registry (Generic API)

```rust
use metallic::caching::CacheRegistry;

let mut registry = CacheRegistry::with_device(device);

// Configure eviction policy (optional)
registry.set_eviction_policy::<MyKernel>(
    EvictionPolicy::size_limited_lru(1000)
);

// Get or create resource
let resource = registry.get_or_create::<MyKernel>(
    &(1024, Dtype::F16, false),
    None,  // Use default device
)?;

// Use the resource
encode_kernel(encoder, &resource.pipeline);
```

#### Via ResourceCache Facade (Optional)

Add a convenience method to `ResourceCache`:

```rust
impl ResourceCache {
    pub fn get_or_create_my_kernel(
        &mut self,
        size: usize,
        dtype: Dtype,
        transpose: bool,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<&mut MyCachedResource, MetalError> {
        self.registry.get_or_create::<MyKernel>(
            &(size, dtype, transpose),
            device,
        )
    }
}
```

Then use it:

```rust
let resource = cache.get_or_create_my_kernel(1024, Dtype::F16, false, None)?;
```

## Eviction Policies

### Overview

Eviction policies prevent unbounded cache growth by automatically removing entries based on configured strategies.

### Available Strategies

#### 1. None (Default)
```rust
EvictionPolicy::none()
```
- No eviction - cache grows unbounded
- Use when: Working set is naturally bounded or memory is not a concern

#### 2. Size-Limited LRU
```rust
EvictionPolicy::size_limited_lru(1000)
```
- Maintains maximum entry count
- Evicts least recently used when full
- Use when: You need strict memory bounds

#### 3. LRU (Manual Limit)
```rust
EvictionPolicy::lru().with_max_entries(500)
```
- Evicts least recently used entry
- Requires explicit size limit
- Use when: You want LRU but with flexible configuration

#### 4. FIFO
```rust
EvictionPolicy::fifo().with_max_entries(500)
```
- Evicts oldest entry by creation time
- Simpler than LRU, ignores access patterns
- Use when: All entries have similar access patterns

#### 5. Idle Timeout
```rust
EvictionPolicy::idle_timeout(Duration::from_secs(300))
```
- Automatically removes entries idle for more than duration
- Runs on every cache operation
- Use when: You want automatic cleanup of stale entries

#### 6. Hybrid (Size + Idle)
```rust
EvictionPolicy::hybrid(1000, Duration::from_secs(300))
```
- Combines size limit with idle timeout
- Both conditions checked independently
- Use when: You want both hard limits and automatic cleanup

### Configuring Eviction

#### Per-Cache Configuration

```rust
// Different policies for different kernels
registry.set_eviction_policy::<MpsGemmKernel>(
    EvictionPolicy::size_limited_lru(1000)
);

registry.set_eviction_policy::<SoftmaxMpsKernel>(
    EvictionPolicy::idle_timeout(Duration::from_secs(300))
);

registry.set_eviction_policy::<SdpaKernel>(
    EvictionPolicy::hybrid(500, Duration::from_secs(600))
);
```

#### Builder Pattern

```rust
let policy = EvictionPolicy::lru()
    .with_max_entries(1000)
    .with_max_idle_duration(Duration::from_secs(300))
    .with_min_entries(10);  // Safety minimum

registry.set_eviction_policy::<MyKernel>(policy);
```

### Eviction Behavior

#### When Eviction Occurs

1. **Cache Hit**: No eviction check (fast path)
2. **Cache Miss**: Eviction runs BEFORE resource creation
3. **Idle Timeout**: Checked on every operation (if configured)

#### What Gets Evicted

- **LRU**: Entry with oldest `last_used_at` timestamp
- **FIFO**: Entry with oldest `created_at` timestamp
- **Idle Timeout**: All entries where `now - last_used_at > max_idle`
- **Size-Limited LRU**: Multiple entries via LRU until under limit

#### Safety Guarantees

- **Min Entries Protection**: `with_min_entries(n)` prevents evicting below n entries
- **Correctness**: Cache hits always update `last_used_at` timestamp
- **Atomicity**: Eviction completes before resource creation

### Performance Characteristics

| Strategy | Eviction Check | Eviction Cost | Notes |
|----------|---------------|---------------|-------|
| None | O(1) | N/A | No overhead |
| LRU | O(1) | O(n) | Scan to find LRU |
| FIFO | O(1) | O(n) | Scan to find oldest |
| IdleTimeout | O(n) | O(k) | Scan all, remove k idle |
| SizeLimitedLru | O(1) | O(m*n) | May evict multiple |

Where:
- n = cache size
- k = number of idle entries
- m = number of entries to evict

**Optimization Notes:**
- Cache hits have zero eviction overhead
- Most strategies evict one entry at a time
- Idle timeout can be disabled for hot paths

## Instrumentation and Metrics

### Automatic Logging

All cache operations are automatically logged via `metallic_instrumentation` when enabled:

#### Cache Access Events

```json
{
  "type": "ResourceCacheAccess",
  "data": {
    "cache_key": "mps_gemm:MpsGemmKey{m:1024,n:1024,k:1024}",
    "hit": true,
    "bytes": 0
  }
}
```

#### Cache Summary (Every 100 ops)

```json
{
  "type": "ResourceCacheSummary",
  "data": {
    "cache": "mps_gemm",
    "hits": 450,
    "misses": 50,
    "hit_rate": 90.0,
    "size": 87
  }
}
```

#### Eviction Events

```json
{
  "type": "CacheEviction",
  "data": {
    "cache": "mps_gemm",
    "strategy": "size_limited_lru",
    "count": 5,
    "reason": "size_limit_exceeded_max_1000",
    "size_after": 1000
  }
}
```

### Querying Metrics

#### Per-Cache Metrics

```rust
use metallic::caching::CacheMetrics;

if let Some(metrics) = registry.metrics::<MpsGemmKernel>() {
    println!("Size: {}", metrics.size);
    println!("Hits: {}", metrics.hits);
    println!("Misses: {}", metrics.misses);
    println!("Evictions: {}", metrics.evictions);
    println!("Hit Rate: {:.2}%", 
        metrics.hits as f64 / (metrics.hits + metrics.misses) as f64 * 100.0
    );
    
    if let Some(age) = metrics.oldest_entry_age {
        println!("Oldest entry: {:?}", age);
    }
}
```

#### All Cache Metrics

```rust
for (name, metrics) in registry.metrics_by_name() {
    println!("{}: {} entries, {} hits, {} misses",
        name, metrics.size, metrics.hits, metrics.misses
    );
}
```

### Metric Fields

```rust
pub struct CacheMetrics {
    pub size: usize,                          // Current entry count
    pub hits: u64,                            // Total hits
    pub misses: u64,                          // Total misses
    pub evictions: u64,                       // Total evictions
    pub last_event: Option<CacheEvent>,       // Most recent event
    pub oldest_entry_age: Option<Duration>,   // Age of oldest entry
    pub newest_entry_age: Option<Duration>,   // Age of newest entry
    pub longest_idle_age: Option<Duration>,   // Longest idle time
    pub shortest_idle_age: Option<Duration>,  // Shortest idle time
    pub max_entry_reuse_count: Option<u64>,   // Highest reuse count
}
```

## Registry Slots (Non-Kernel Caches)

For caches that don't fit the kernel pattern (e.g., KV cache allocations, tensor preparation):

### Implementing a Slot

```rust
use metallic::caching::CacheRegistrySlot;

struct MyContextCache {
    data: FxHashMap<String, MyData>,
}

impl CacheRegistrySlot for MyContextCache {
    fn clear_slot(&mut self) {
        self.data.clear();
    }
}
```

### Using Slots

```rust
// Initialize or get existing slot
let cache = registry.slot_mut(|| MyContextCache {
    data: FxHashMap::default(),
});

// Use the cache
cache.data.insert("key".to_string(), my_data);

// Clear slot
registry.clear_slot::<MyContextCache>();
```

**Slot Use Cases:**
- KV cache allocation tracking (`KvCacheState<T>`)
- Tensor preparation optimization (`TensorPreparationCache<T>`)
- Per-context state that doesn't fit the kernel pattern

## Best Practices

### Cache Key Design

✅ **DO:**
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GoodKey {
    m: usize,
    n: usize,
    k: usize,
    dtype: Dtype,       // Enum (cheap copy)
    transpose: bool,
}
```

❌ **DON'T:**
```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BadKey {
    matrix_data: Vec<f32>,     // Too much data
    timestamp: Instant,         // Non-deterministic
    name: String,               // Unnecessary allocation
}
```

### Eviction Policy Selection

| Scenario | Recommended Policy | Reasoning |
|----------|-------------------|-----------|
| Small working set (<100 entries) | `None` | No eviction overhead |
| Large working set (>1000 entries) | `size_limited_lru(1000)` | Prevent OOM |
| Variable access patterns | `lru().with_max_entries(500)` | Optimize for temporal locality |
| Predictable lifecycles | `fifo().with_max_entries(500)` | Simpler, lower overhead |
| Long-running processes | `idle_timeout(Duration::from_secs(300))` | Auto cleanup |
| Production deployment | `hybrid(1000, Duration::from_secs(600))` | Belt and suspenders |

### Size Limit Guidelines

```rust
// Conservative (recommended for production)
EvictionPolicy::size_limited_lru(1000)

// Aggressive (for memory-constrained environments)
EvictionPolicy::size_limited_lru(100)
    .with_min_entries(10)  // Always keep minimum

// Generous (for development/debugging)
EvictionPolicy::size_limited_lru(10000)
```

**Rule of Thumb:**
- Start with 1000 entries per cache
- Monitor hit rates and evictions via metrics
- Tune based on actual usage patterns
- Use `min_entries` to prevent thrashing

### Error Handling

✅ **DO:**
```rust
let resource = registry.get_or_create::<MyKernel>(&params, device)?;
// Propagate errors properly
```

❌ **DON'T:**
```rust
let resource = registry.get_or_create::<MyKernel>(&params, device)
    .unwrap();  // Can panic in production!
```

### Threading

**Important:** `CacheRegistry` is **NOT thread-safe** by design.

✅ **DO:**
```rust
// Per-context registry (typical usage)
struct Context<T> {
    cache_registry: CacheRegistry,  // Owned by context
    // ...
}

// One context per thread (recommended pattern)
```

❌ **DON'T:**
```rust
// Sharing registry across threads requires external synchronization
static REGISTRY: Mutex<CacheRegistry> = ...;  // Avoid if possible
```

**Why:** Caches are typically scoped to a Metal command buffer/encoder lifecycle, which is inherently single-threaded.

## Advanced Topics

### Cache Warming

Pre-populate caches with known configurations:

```rust
fn warm_cache(registry: &mut CacheRegistry, device: &Device) -> Result<(), MetalError> {
    // Common sizes
    let sizes = [256, 512, 1024, 2048, 4096];
    let dtypes = [Dtype::F16, Dtype::F32];
    
    for &size in &sizes {
        for &dtype in &dtypes {
            registry.get_or_create::<MpsGemmKernel>(
                &(size, size, size, dtype, false, false),
                Some(device),
            )?;
        }
    }
    
    Ok(())
}
```

### Dynamic Policy Adjustment

Adjust policies at runtime based on metrics:

```rust
fn tune_eviction(registry: &mut CacheRegistry) {
    if let Some(metrics) = registry.metrics::<MpsGemmKernel>() {
        let hit_rate = metrics.hits as f64 / (metrics.hits + metrics.misses) as f64;
        
        if hit_rate < 0.80 {
            // Low hit rate - increase cache size
            registry.set_eviction_policy::<MpsGemmKernel>(
                EvictionPolicy::size_limited_lru(2000)
            );
        } else if metrics.evictions > 1000 {
            // Too many evictions - increase size
            registry.set_eviction_policy::<MpsGemmKernel>(
                EvictionPolicy::size_limited_lru(1500)
            );
        }
    }
}
```

### Custom Eviction Strategies

If built-in strategies don't fit, implement a custom wrapper:

```rust
struct CustomEvictionCache<K: CacheableKernel> {
    inner: CacheRegistry,
    custom_logic: Box<dyn Fn(&CacheMetrics) -> bool>,
}

impl<K: CacheableKernel> CustomEvictionCache<K> {
    pub fn get_or_create(
        &mut self,
        params: &K::Params,
        device: Option<&Device>,
    ) -> Result<&mut K::CachedResource, MetalError> {
        // Check custom condition
        if let Some(metrics) = self.inner.metrics::<K>() {
            if (self.custom_logic)(&metrics) {
                self.inner.clear();  // Custom eviction
            }
        }
        
        self.inner.get_or_create::<K>(params, device)
    }
}
```

## Troubleshooting

### High Miss Rate

**Symptom:** Cache hit rate < 80%

**Possible Causes:**
1. Cache keys too granular (each request unique)
2. Eviction policy too aggressive
3. Working set larger than cache size

**Solutions:**
```rust
// 1. Use bucketing in cache keys
fn create_cache_key(params: &Self::Params) -> Self::Key {
    MyKey {
        // Round to nearest 256
        size: (params.size + 255) / 256 * 256,
        dtype: params.dtype,
    }
}

// 2. Increase cache size
registry.set_eviction_policy::<MyKernel>(
    EvictionPolicy::size_limited_lru(2000)  // Was 1000
);

// 3. Check metrics to understand patterns
let metrics = registry.metrics::<MyKernel>()?;
eprintln!("Size: {}, Evictions: {}", metrics.size, metrics.evictions);
```

### Memory Growth

**Symptom:** Memory usage grows unbounded

**Cause:** No eviction policy configured

**Solution:**
```rust
// Set eviction for all kernel caches
registry.set_eviction_policy::<MpsGemmKernel>(
    EvictionPolicy::size_limited_lru(1000)
);
registry.set_eviction_policy::<SoftmaxMpsKernel>(
    EvictionPolicy::size_limited_lru(500)
);
// ... etc
```

### Thrashing

**Symptom:** High eviction count, low hit rate, poor performance

**Cause:** Cache size smaller than working set

**Solutions:**
```rust
// Increase cache size
registry.set_eviction_policy::<MyKernel>(
    EvictionPolicy::size_limited_lru(5000)
        .with_min_entries(100)  // Prevent over-eviction
);

// Or use idle timeout instead
registry.set_eviction_policy::<MyKernel>(
    EvictionPolicy::idle_timeout(Duration::from_secs(300))
);
```

### Slow Cache Misses

**Symptom:** First-time operations very slow

**Cause:** Resource creation is expensive (expected)

**Solutions:**
1. Cache warming (pre-populate common cases)
2. Async compilation (if possible)
3. Increase cache size to reduce misses

```rust
// Warm cache at startup
warm_cache(&mut registry, &device)?;

// Or show progress
let start = Instant::now();
let resource = registry.get_or_create::<MyKernel>(&params, device)?;
if start.elapsed() > Duration::from_millis(10) {
    eprintln!("Cache miss took {:?}", start.elapsed());
}
```

## Migration Guide

### From Legacy Cache System

If migrating from the old `cacheable_resources/*` system:

1. **Move cache key to kernel module:**
```rust
// OLD: crates/metallic/src/cacheable_resources/mps_gemm.rs
// NEW: crates/metallic/src/kernels/matmul_mps/cache.rs
```

2. **Implement CacheableKernel:**
```rust
impl CacheableKernel for MpsGemmKernel {
    type Key = MpsGemmKey;           // Was separate
    type CachedResource = MpsGemm;   // Was separate
    type Params = (...);
    // ...
}
```

3. **Update call sites:**
```rust
// OLD
let resource = get_cached_mps_gemm(key, device)?;

// NEW
let resource = registry.get_or_create::<MpsGemmKernel>(&params, device)?;
```

4. **Remove old cache logic:**
- Delete `cacheable_resources/` directory
- Delete `cache_keys/` directory
- Remove manual cache management code

## Summary

The metallic caching infrastructure provides:

✅ **Type-safe** - Compile-time guarantees, no runtime type errors  
✅ **Performant** - Zero-cost abstractions, fast hash maps  
✅ **Flexible** - Per-cache eviction policies  
✅ **Observable** - Full JSONL instrumentation  
✅ **Maintainable** - Clean separation of concerns  
✅ **Extensible** - Easy to add new kernel types  

For most use cases:
1. Implement `CacheableKernel` for your kernel
2. Set `EvictionPolicy::size_limited_lru(1000)` in production
3. Monitor metrics via JSONL logs
4. Tune based on actual usage patterns

**Next Steps:**
- See `crates/metallic/src/kernels/matmul_mps/cache.rs` for complete example
- See `crates/metallic/src/caching/eviction_tests.rs` for usage patterns
- See `metallic_instrumentation` docs for metric collection
