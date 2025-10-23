# MPSGraph SDPA Integration Status

This document tracks the integration of MPSGraph-based Scaled Dot-Product Attention (SDPA) into the Metallic kernel system, the current implementation state, masking assumptions, and the remaining work items.

## Goals
- Provide an MPSGraph backend for SDPA as an alternative to our existing composite implementation (QK^T → softmax → V) to evaluate performance and simplify maintenance.
- Initial support is f16-only; f32 support will be introduced later.
- Support causal masking with an additive mask convention (0 for allowed, −inf for disallowed). Note: mask semantics are assumed and may vary by OS/version; we will validate parity empirically and adjust if needed.

## Current Implementation

- Module introduced: `crates/metallic/src/kernels/sdpa_mps_graph/mod.rs`
  - Exposes `SdpaMpsGraphOp` implementing `KernelInvocable`.
  - Performs shape/dtype validation and enforces f16-only for now (`todo!()` for f32).
  - Resource cache now builds and stores a fully compiled `MPSGraphExecutable`, ordered feed/target tensors, tensor shapes, and a pooled causal-mask `MTLBuffer` per `(batch, seq_q, seq_k, dim, causal, dtype)` key.
  - Execution path:
    - Ends any active compute encoder, wraps the in-flight `CommandBuffer` in an `MPSCommandBuffer`, and encodes the executable with `encodeToCommandBuffer:inputsArray:resultsArray:executionDescriptor:`.
    - Binds Q/K/V (and cached mask when causal) directly from their `MTLBuffer`s; the preallocated output tensor is passed as a result buffer so the graph writes in-place with no CPU staging or blit.
    - Feed/target layouts are derived from the executable metadata and kept alongside the cache entry so we can extend the graph with additional nodes later without reworking the encode path.

- Benchmarks:
  - New bench: `benches/sdpa_mpsgraph_vs_current_benchmark.rs` compares MPSGraph SDPA vs current optimized SDPA for a Qwen-like shape at f16, causal=true.
  - Existing bench: `benches/sdpa_variant_benchmark.rs` remains available for broader SDPA variant comparison (f32-focused today).

- Dependencies:
  - `objc2-metal-performance-shaders-graph` added to `crates/metallic/Cargo.toml`.
  - The project already uses `objc2-metal`, `objc2-foundation`, and `objc2-metal-performance-shaders`.

## Masking Assumptions
- We assume additive mask semantics:
  - Allowed positions: 0.0
  - Disallowed positions: −inf
- If the MPSGraph API uses a different convention (e.g., boolean masks), we will adapt after parity tests.
- The code contains comments calling out this assumption and the need to verify via tests/benches.

## Remaining Work

1) f32 enablement  
   - Extend the cache builder to emit Float32 placeholders, mask buffers, and compile-time descriptors once we are ready to validate accuracy and performance for fp32 models.

2) Larger graph segments  
   - Experiment with appending normalization/projection nodes around SDPA inside the same `MPSGraph` so we can amortize execution costs and reduce Metal encoder churn even further.

3) Runtime controls and telemetry  
   - Add `METALLIC_FORCE_SDPA_BACKEND=mpsgraph` (and possibly percentage-based rollout hooks) plus cache hit/miss reporting for the new executable/mask pools.

4) Dynamic-mask reuse  
   - Investigate bucketing strategies so a single cached mask can serve multiple `(seq_q, seq_k)` shapes during decode growth without reallocation while keeping correctness guarantees.

## Filepaths and References
- Entry point:
  - `crates/metallic/src/kernels/sdpa_mps_graph/mod.rs`
- Support systems:
  - Context: `crates/metallic/src/context.rs` (ending encoders, profiling, command queue access)
  - Resource cache: `crates/metallic/src/resource_cache.rs` (for future executable/mask caching)
  - Softmax-MPS implementation reference: `crates/metallic/src/kernels/softmax_mps/mod.rs`
- Benchmarks:
  - `benches/sdpa_mpsgraph_vs_current_benchmark.rs`
  - `benches/sdpa_variant_benchmark.rs`

## Observations
- `MPSGraphExecutable::feedTensors` / `targetTensors` give us an authoritative ordering, which we now retain in the cache; this should make future graph growth straightforward without touching the encode path again.
- Re-wrapping the active `MTLCommandBuffer` via `MPSCommandBuffer::commandBufferWithCommandBuffer` keeps execution inside the caller’s command buffer timeline and preserves profiling hooks.
- Zero-copy bindings highlighted that masks must live in shared memory today; if we need private storage later we will have to rely on explicit synchronization or new API surface from Apple.
- Mask semantics remain additive 0/−inf; parity validation across OS releases is still required before we flip the switch broadly.

## Milestone C Updates
- New `GraphKernel` trait advertises storage/accumulator policy so graph-friendly kernels can share cache keys and validate dtype support defensively.
- `GraphKernel::signature()` now exposes per-binding axis semantics so future kernels can reuse the SDPA descriptors when wiring additional graph builders.
- `KernelBackendRegistry` now mediates SDPA backend selection, honoring `METALLIC_FORCE_SDPA_BACKEND=legacy|mpsgraph|auto` and emitting `KernelBackendSelected` metrics for observability.
- The CLI exposes `--sdpa-backend {auto|legacy|graph}` which maps onto the registry overrides, enabling per-run experimentation without touching env vars.
- SDPA dispatch uses the registry by default, allowing Qwen25 pipelines to toggle graph execution without code changes.
- Graph caches have been refactored into `GraphExecutableCache` and `MaskArena`, preserving instrumentation while enabling reuse by upcoming layernorm/projection ports.
- Cache metrics include lifetime/eviction signals (oldest/newest entry age, idle windows, reuse counts) to guide future eviction policy tuning and dashboard surfacing.
- SDPA now compiles dynamic MPSGraph executables; placeholders accept runtime batch/sequence lengths and bindings provide the concrete shapes on each call, so incremental decode stays on the graph path without per-step recompiles.

## Next Steps

## Cache Telemetry and Metrics
The MPSGraph SDPA implementation integrates with the `metallic_instrumentation` system for comprehensive cache monitoring:

- **Cache Access Metrics**: Each resource cache operation emits `MetricEvent::ResourceCacheAccess` events that capture:
  - `cache_key`: A descriptive key showing the cache type and resource parameters
  - `hit`: Boolean indicating hit (true) or miss (false)
  - `bytes`: Placeholder value (0) that can be extended to track resource sizes

- **Periodic Summary Metrics**: Every 100 cache operations, the system emits `MetricEvent::ResourceCacheSummary` events with:
  - `cache`: Cache name (e.g., "mpsgraph_sdpa", "gemm", "softmax", etc.)
  - `hits`: Total cache hits count
  - `misses`: Total cache misses count  
  - `hit_rate`: Cache hit rate as percentage (0-100)
  - `size`: Current cache size in number of entries

- **Compilation and Execution Timing**: The implementation captures performance metrics for:
  - `sdpa_compile`: Time taken to compile MPSGraph executable
  - `sdpa_encode`: Time taken to encode execution on the command buffer

1) Run the sdpa benchmarks (and new parity tests) to quantify latency vs. the legacy kernel now that we avoid the staging copy.
2) Prototype an extended graph (e.g., SDPA + output projection) to validate that the cached feed/target wiring handles multi-node executables cleanly.
3) Land the backend toggle and cache telemetry so we can selectively roll out the MPSGraph path and monitor hit rates in production workloads.
4) Explore mask bucketing or incremental-updates to support streaming decode without regenerating large masks each step.

## Validation / Testing Plan
- Compare outputs vs. existing SDPA for a matrix of shapes:
  - Batches: 1, 4, 8
  - Seq: 128, 256, 512
  - Dim: 64, 128
  - Causal true/false
- Ensure tolerances match f16 numerical ranges.
- Profile with `metallic_instrumentation` under latency mode to capture end-to-end timings.

## Known Gaps
- No f32 support yet (intentional until we validate tolerances).
- Mask cache is exact shape keyed; we still recreate entries for every new `(seq_q, seq_k)` pair instead of reusing buckets.
- Graph today only contains SDPA; larger fused segments still need prototyping and heuristics for when to select them.

## Tracking
- Consider creating Jira item: “Validate MPSGraph SDPA mask semantics and parity with existing SDPA (causal)”
- Consider Confluence page summarizing this status and linking to this file, with benchmark results once available.
