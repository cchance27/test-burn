# Metallic Engine Improvement Backlog

This checklist captures potential optimizations and hardening opportunities discovered during the review. Items are grouped by expected effort/complexity, but many can be pursued in parallel.

## ✅ Quick wins
- [x] Record the dtype and element size in `MemoryPool::alloc_tensor` instead of assuming `f32`; this prevents over-allocation and unlocks pooled storage for `bf16`/`fp16` tensors. (Implemented via `PooledAllocation` metadata on the dedicated KV pool.)
- [x] Expose pooled tensor allocation for KV cache setup to avoid the extra `clone()` handles that retain buffers longer than necessary. (KV initialization now requests typed pooled buffers directly.)
- [x] Remove unnecessary `clone()` calls when dispatching operations (e.g., `Context::matmul*`); prefer borrowed tensor handles or lightweight views to cut reference counting churn. (Dispatch helpers now route borrowed tensor arguments through `KernelInvocable::Args`, only cloning when command recording requires ownership, and the kernel tests exercise the borrowed contract.)
- [x] Replace repeated manual F16→F32 conversion paths with a single helper and skip debug prints to reduce temporary allocations during GGUF loading. 【F:src/gguf/model_loader.rs†L27-L136】
- [x] Audit `ResourceCache::get_or_create_resource`’s dummy-device `transmute` and replace with an explicit optional device handle to avoid UB and simplify lifetime tracking. 【F:src/Metallic/resource_cache.rs†L34-L73】 (Resource cache now accepts explicit device handles via `with_device` constructor and properly handles optional device parameters in `Cacheable::from_key`, eliminating the unsafe dummy-device transmute.)
- [ ] Add debug assertions (enabled in non-release builds) that ensure borrowed tensor arguments outlive the command buffer submission boundaries now that dispatch helpers avoid cloning. 【F:src/Metallic/context.rs†L60-L161】
- [x] Plumb explicit synchronization for tensors returned from pooled allocations (e.g., mark when blit zeroing completes) so downstream code can skip redundant `clone()` handles. (KV cache entries track a zeroing-complete flag consumed by fast paths.)

## ⚙️ Medium-effort improvements
- [ ] Switch pooled and ad-hoc tensors from `StorageModeShared` to `StorageModePrivate` with blit-based staging buffers to cut host RAM footprint and improve bandwidth. 【F:src/Metallic/pool.rs†L98-L107】【F:src/Metallic/tensor.rs†L78-L131】
- [ ] Introduce a dedicated host staging allocator (possibly using `MTLHeap` or `MTLBuffer` recycling) to avoid allocating new shared buffers for every GGUF tensor. 【F:src/gguf/model_loader.rs†L20-L135】
- [ ] Add lazy, demand-paged GGUF tensor loading (memory-mapped files plus per-layer uploads) instead of materializing the full model as `Vec<f32>` in RAM. 【F:src/gguf/model_loader.rs†L35-L135】
- [ ] Expand `Tensor` to carry dtype metadata through kernels and loaders so models can stay in native precision (e.g., keep weights in `bf16/fp16` until a kernel demands `fp32`). 【F:src/Metallic/tensor.rs†L19-L140】
- [ ] Introduce fused compute passes (e.g., matmul + bias + activation) where the intermediate results can stay in threadgroup memory, reducing buffer churn and GPU↔CPU sync. 【F:src/Metallic/kernels】【F:src/Metallic/context.rs†L60-L169】
- [ ] Track pool utilization metrics (peak/average) and expose them for regression detection; wire into `Context` metrics to spot leaks. 【F:src/Metallic/context.rs†L18-L57】【F:src/Metallic/pool.rs†L12-L110】
- [ ] Reuse KV cache allocations between runs by resetting ranges instead of dropping and reallocating buffers, and expose statistics to monitor growth. 【F:src/Metallic/context.rs†L117-L169】
- [ ] Implement tensor views / slices that reuse the same buffer with offset/stride metadata instead of creating full copies when reshaping or selecting timesteps. 【F:src/Metallic/tensor.rs†L48-L77】
- [ ] Add fallbacks for compact attention (paged KV, sliding window) so long-context runs don’t require enormous contiguous buffers. 【F:src/Metallic/context.rs†L117-L169】

## 🚀 Longer-term upgrades
- [ ] Integrate mixed-precision kernels (FP16/BF16) alongside FP32 to shrink bandwidth and memory pressure; gate on device capabilities. 【F:src/Metallic/tensor.rs†L19-L140】【F:src/Metallic/kernels】
- [ ] Support per-layer offload strategies (weights in memory-mapped CPU space, activations/KV in VRAM) with configurable placement policies. 【F:src/gguf/model_loader.rs†L20-L135】【F:src/Metallic/context.rs†L18-L169】
- [ ] Move pooled allocations onto `MTLHeaps` to enable sub-allocations with explicit residency management and defragmentation hooks. 【F:src/Metallic/pool.rs†L12-L110】
- [ ] Develop a background residency manager that migrates rarely used tensors back to host (or compresses them) when pool pressure is high. 【F:src/Metallic/context.rs†L18-L169】
- [ ] Introduce per-op command-buffer recycling and reuse to prevent the active command buffer from holding onto large caches longer than required. 【F:src/Metallic/context.rs†L34-L91】
- [ ] Add comprehensive profiling hooks (Metal counters, custom timers) to identify kernel-level stalls responsible for the observed <10 it/s throughput. 【F:src/Metallic/context.rs†L34-L100】
- [ ] Extend GGUF loader to support quantized tensor formats (Q4/Q6/etc.) natively on Metal, eliminating CPU-side dequantization expansions. 【F:src/gguf/model_loader.rs†L20-L135】
- [ ] Port scaled-dot-product attention (including FlashAttention-style tiling) into a dedicated Metal compute pipeline so attention stays resident on-device and benefits from fused softmax + matmul kernels. 【F:src/Metallic/kernels】【F:src/Metallic/context.rs†L95-L169】

> **Testing reminder:** The sandbox lacks Apple Silicon/Metal support, so please run performance and functional tests locally on a Metal-enabled machine.
