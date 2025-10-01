# Metallic Engine Improvement Backlog

This checklist captures potential optimizations and hardening opportunities discovered during the review. Items are grouped by expected effort/complexity, but many can be pursued in parallel.

## ✅ Quick wins
- [ ] Add debug assertions (enabled in non-release builds) that ensure borrowed tensor arguments outlive the command buffer submission boundaries now that dispatch helpers avoid cloning. 【F:src/metallic/context.rs†L60-L161】

## ⚙️ Medium-effort improvements
- [ ] Add lazy, demand-paged GGUF tensor loading (memory-mapped files plus per-layer uploads) instead of materializing the full model as `Vec<f32>` in RAM. 【F:src/gguf/model_loader.rs†L35-L135】
- [ ] Introduce fused compute passes (e.g., matmul + bias + activation) where the intermediate results can stay in threadgroup memory, reducing buffer churn and GPU↔CPU sync. 【F:src/metallic/kernels】【F:src/metallic/context.rs†L60-L169】
- [ ] Track pool utilization metrics (peak/average) and expose them for regression detection; wire into `Context` metrics to spot leaks. 【F:src/metallic/context.rs†L18-L57】【F:src/metallic/pool.rs†L12-L110】
- [ ] Reuse KV cache allocations between runs by resetting ranges instead of dropping and reallocating buffers, and expose statistics to monitor growth. 【F:src/metallic/context.rs†L117-L169】
- [ ] Implement tensor views / slices that reuse the same buffer with offset/stride metadata instead of creating full copies when reshaping or selecting timesteps. 【F:src/metallic/tensor.rs†L48-L77】
- [ ] Add fallbacks for compact attention (paged KV, sliding window) so long-context runs don’t require enormous contiguous buffers. 【F:src/metallic/context.rs†L117-L169】

## 🚀 Longer-term upgrades
- [ ] Support per-layer offload strategies (weights in memory-mapped CPU space, activations/KV in VRAM) with configurable placement policies. 【F:src/gguf/model_loader.rs†L20-L135】【F:src/metallic/context.rs†L18-L169】
- [ ] Move pooled allocations onto `MTLHeaps` to enable sub-allocations with explicit residency management and defragmentation hooks. 【F:src/metallic/pool.rs†L12-L110】
- [ ] Develop a background residency manager that migrates rarely used tensors back to host (or compresses them) when pool pressure is high. 【F:src/metallic/context.rs†L18-L169】
- [ ] Introduce per-op command-buffer recycling and reuse to prevent the active command buffer from holding onto large caches longer than required. 【F:src/metallic/context.rs†L34-L91】
- [ ] Add comprehensive profiling hooks (Metal counters, custom timers) to identify kernel-level stalls responsible for the observed <10 it/s throughput. 【F:src/metallic/context.rs†L34-L100】
- [ ] Extend GGUF loader to support quantized tensor formats (Q4/Q6/etc.) natively on Metal, eliminating CPU-side dequantization expansions. 【F:src/gguf/model_loader.rs†L20-L135】
- [ ] Port scaled-dot-product attention (including FlashAttention-style tiling) into a dedicated Metal compute pipeline so attention stays resident on-device and benefits from fused softmax + matmul kernels. 【F:src/metallic/kernels】【F:src/metallic/context.rs†L95-L169】

> **Testing reminder:** The sandbox lacks Apple Silicon/Metal support, so please run performance and functional tests locally on a Metal-enabled machine.
