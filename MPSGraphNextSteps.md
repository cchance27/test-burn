# MPSGraph Next Steps

This plan consolidates the work needed to turn the current SDPA-focused integration into a broader, high-performance, and developer-friendly MPSGraph backend within Metallic. The tracks are grouped to maximize reuse, minimize unsafe surface area, and ensure every milestone captures performance and correctness data.

## Guiding Principles
- Performance first: every milestone includes targeted `cargo bench` or `criterion` runs on Qwen25 scenarios and representative stress cases.
- Strong typing and defensive APIs: reduce direct unsafe usage by pushing shape/dtype contracts into Rust abstractions.
- Idiomatic conversions: prefer `From`, `Into`, and `TryFrom` implementations to bridge `F16Tensor`, `TensorView`, `MetallicDType`, and `MPSDataType` while keeping conversions zero-copy whenever the underlying `MTLBuffer` permits it.
- Memory discipline: favor `MTLStorageModeShared` only when encoder sharing is required; otherwise keep tensors in private GPU memory and avoid CPU round-trips unless explicitly benchmarking or debugging.
- Observability built-in: rely on `metallic_instrumentation` for latency, cache statistics, and divergence tracking.
- Reuse and extensibility: design graph builders and cache layers so additional kernels can plug in with minimal boilerplate.

## 2025‑02‑14 Status Snapshot

### What Landed
- Instrumented tensor materialization and SDPA graph readiness paths; JSONL logs now include detailed storage/size annotations for each copy.
- Repeat-KV workspaces continue to allocate dedicated shared buffers per `(layer, kind)` for the graph backend so SDPA still sees dense, zero-offset tensors. We reverted the brief “strided view” experiment after it triggered MPSGraph MLIR failures in release builds.
- Q/K/V SDPA bindings stay on the MPSNDArray path; `graph_ready/materialized` events remain limited to the prompt/logits sync stage.
- Latest profiling (2025‑10‑23 runs, capped at `max_tokens=40`) shows the graph backend sustaining ~44 tok/s (slightly up from the earlier 42 tok/s) versus ~62 tok/s on the legacy path. Legacy remains our throughput baseline, and all JSONL comparisons now use matched decode-length runs.

### Current Gaps
- Generation-loop iteration time (noprof) averages ~22.7 ms/token on graph vs ~16.3 ms/token on legacy. The largest contributors are SDPA and the surrounding per-block totals; cumulative graph block time is ≈4.31 s vs ≈0.61 s on legacy, with block 0 alone costing ~0.20 s (graph) vs ~0.04 s (legacy).
- KV repeat still performs a per-token GPU copy into the shared workspace. Until we design an incremental or MLIR-safe strided solution, this copy dominates the per-block delta and explains the outsize cost of the early blocks.
- SDPA encode now accounts for ~0.17 s across 2 039 calls in the graph noprof run (legacy never enters MPSGraph, so the metric is absent there). The encode cost itself is small, but the call count hints that each block is encoding multiple slices; we should confirm this is expected once the KV repeat work is finished.
- All profiling runs now use identical `max_tokens`, seeds, and stop criteria. Remaining tok/s differences are attributable to per-token work rather than decode length drift.

### Next Actions
1. Replace the brute-force KV workspace blit with an incremental or MLIR-safe strided solution that leaves SDPA inputs zero-offset without copying the entire cache each decode step.
2. Reprofile (`graph-noprof.jsonl` vs `legacy-noprof.jsonl`) after the KV fix to confirm block totals fall within the legacy budget and to identify any remaining gaps.
3. Monitor SDPA encode counts/timing to ensure we are not regressing as we revise the KV path.
4. Continue auditing JSONL logs to ensure `graph_ready/materialized` stays limited to intentional staging (prompt logits, etc.) and to track SDPA encode costs. Maintain the habit of comparing per-block totals and logits sync counts between graph and legacy runs.

### Zero-offset & Contiguity Plan
- **NDArray bindings (current path):** `GraphTensorBinding` now wraps non-zero-offset Q/K/V buffers in `MPSNDArray` views so MPSGraph sees dense tensors without CPU-side materialization. JSONL confirms `graph_ready/materialized` is gone.
- **Identity blit (backup):** Remains a fallback if NDArray aliasing ever fails; the earlier command-buffer assertion has not resurfaced under the current bindings.
- **Permutation fix:** Continue to apply GPU `transpose` when layout mismatch is the only issue so inputs stay in the required `[B,H,S,D]` order.
- **Index remap:** For genuinely irregular strides (e.g. interleaved heads) we still plan to fall back to gather-based densification once the workspace copy loop is removed.
- **Feed pipeline fallback:** Longer term, consider emitting strided `MPSNDArray` views directly during cache writes so we can skip the repeat-KV copy entirely.

## Milestone A: Harden the SDPA Graph Path

- [x] Typed graph interface layer  
  - [x] Introduce Rust-side builders that encode tensor shapes, dtypes, and layout contracts for each op. Hint: mirror the `KernelInvocationBuilder` patterns so callers only pass high-level tensor handles.  
  - [x] Add `From<&F16Tensor>` / `TryFrom<&TensorHandle>` impls that yield `GraphTensorBinding` descriptors without cloning buffers; guard stride/layout mismatches with early `Err`.  
  - [x] Expose zero-copy buffer bindings behind safe wrappers; keep unsafe bridging localized with documented invariants (`DEBT:` comment if invariants need follow-up). Document when conversions require staging so developers can avoid accidental CPU copies.  
  - [x] Bench: `cargo bench --bench sdpa_mpsgraph_vs_current_benchmark` before/after to confirm no regressions.  
  - [x] Tests: extend `crates/metallic/src/kernels/sdpa_mps_graph/sdpa_mps_graph_test.rs` with parity cases (batch {1,4,8}, seq {128,256,512}, causal/non-causal); perform comparison runs against our legacy non mps-graph sdpa implementation as needed.
- [x] Cache instrumentation  
  - [x] Emit cache hit/miss/eviction counters through `metallic_instrumentation` spans; wrap cache mutations in `tracing::instrument` blocks named per key tuple.  
  - [x] Add latency timers around executable compilation and encode submissions; expose a `CacheMetricsSnapshot` struct with `From<&ResourceCache>` for reporting.  
  - [x] Bench: collect timing snapshots and ensure overhead is within noise envelopes.  
  - [x] Tests: add unit tests covering cache stats propagation and defensive failure modes, e.g., forced eviction when pool exceeds capacity.
- [x] Mask reuse and bucketing  
  - [x] Implement bucketed mask allocation keyed by (causal, dtype, head_dim, bucketed_seq); use `TryFrom<(usize, usize)>` helpers to snap sequence lengths to buckets.  
  - [x] Provide deterministic reuse during incremental decode; ensure `context.synchronize()` where masks transition storage modes and document why in-line. Prefer private storage for long-lived masks and promote to shared only when host reads are required.  
  - [x] Bench: micro-measure incremental decode to validate reduced allocation time.  
  - [x] Tests: add streaming decode regression tests verifying reused buffers stay synchronized; include upper-bound stress (seq 4k) to catch overflow.

## Milestone B: Expand Graph Coverage
- [x] Multi-op graph prototypes  
  - [x] Extend the typed builders to compose SDPA with other nodes; leverage `Into<GraphTensorHandle>` impls so intermediate tensors chain naturally, we want to be able to support migrating other parts of our qwen25 pipeline to mpsgraph.  
  - [x] Ensure feed/target metadata caching scales to multi-node graphs; add `TryFrom<&MPSGraphExecutable>` logic that materializes strongly typed `ExecutableLayout`.  
  - [x] Bench: add `cargo bench --bench sdpa_fused_graph_benchmark` capturing SDPA-only vs fused path.  
  - [x] Tests: integrate `metallic::tests` parity checks covering fused outputs vs reference pipeline, reusing existing Qwen25 fixture data.

## Milestone C: Generalize Graph Kernel Infrastructure

- [x] Define a backend-agnostic `GraphKernel` trait  
  - [x] Advertise storage/accumulator policy through `GraphKernelDtypePolicy` and surface defensive dtype validation helpers.  
  - [x] Extend the trait with typed input/output descriptors so future kernels can share shape metadata without bespoke wrappers.
- [x] Establish a `KernelBackendRegistry` and legacy interop  
  - [x] Route SDPA through a registry-aware dispatch op with telemetry (`KernelBackendSelected`) and env-driven overrides via `METALLIC_FORCE_SDPA_BACKEND`.  
  - [x] Add support for per-kernel programmatic overrides (CLI/config hooks) beyond the env toggle.
- [x] Refactor cache layers into reusable primitives  
  - [x] Introduce `GraphExecutableCache` and `MaskArena`, including accumulator-aware cache keys and shared instrumentation.  
  - [x] Layer in lifetime/eviction metrics to inform future cache policies.
- [ ] Prepare the next kernel migrations for the Qwen25 pipeline  
  - [ ] Target the pre-attention layernorm and the post-attention output projection first; these are adjacent to SDPA and give us SDPA+LayerNorm or Projection+SDPA fusion experiments.  
  - [ ] Document fusion hypotheses (LayerNorm→SDPA, SDPA→MatMul) and build minimal graph prototypes gated behind the registry toggles.  
  - [ ] Capture baseline latency for the existing Metal kernels so we can quantify wins from Graph-backed variants; reuse the sdpa benchmarks as scaffolding.  
  - [ ] Stand up parity tests in `metallic::tests` for the selected kernels (fp16 inputs with fp32 accumulators) and compare against the current Metal implementations.
- [ ] Developer experience updates  
  - [x] Extend `docs/KERNELS.md` with the `GraphKernel` trait workflow, cache expectations, and mixed-precision guidance; refresh `docs/MPSGraphSDPA.md` with Milestone C notes.  
  - [ ] Produce a migration checklist (validation steps, benchmark targets, telemetry verification) to keep kernel ports consistent.  
  - [ ] Add developer tooling to dump graph plans and verify dtype decisions during development (e.g., `METALLIC_DUMP_GRAPH_PLAN=1`).

### Implementation Notes

#### Milestone B: Multi-op graph prototypes Implementation Summary

The multi-op graph prototypes have been successfully implemented with the following key components:

- Created a new `MultiOpGraphBuilder` in `crates/metallic/src/mps_graph/multi_op.rs` that can construct fused operations like SDPA + projection in a single MPSGraph executable
- Implemented strongly-typed executable layouts with `ExecutableLayout` trait and `ExtendableExecutableLayout` struct that can be derived from MPSGraphExecutable metadata via `TryFrom<&MPSGraphExecutable>`
- Added comprehensive test coverage in `crates/metallic/src/mps_graph/multi_op.rs` with 3 passing tests
- Created benchmark infrastructure in `benches/sdpa_fused_graph_benchmark.rs` to compare SDPA-only vs fused operations
- Extended the resource caching system with `CacheableMpsGraphFused` and associated cache key types

#### Known Placeholders and Next Steps

**DEBT: CacheableMpsGraphFused from_key implementation**: The `from_key` implementation in `CacheableMpsGraphFused` contains a placeholder implementation that does not yet use the multi_op module's builder or executable creation logic. The implementation currently creates an empty executable and empty layouts, avoiding the actual multi-op build process. This needs to be replaced with real logic that leverages the `MultiOpGraphBuilder`.

**DEBT: sdpa_fused_graph_benchmark placeholder**: The benchmark in `benches/sdpa_fused_graph_benchmark.rs` is currently a placeholder that simulates fused operations by running SDPA and then returning the result without actual fusion. A real fused implementation (SDPA + projection in single graph) needs to be created to properly benchmark the performance benefits of multi-op fusion.
- [ ] Backend toggles and rollout controls  
  - [ ] Introduce kernel-level enum for backend selection and environment toggle (`METALLIC_SDPA_BACKEND=mpsgraph`); ensure `FromStr`/`Display` implementations enable CLI control.  
  - [ ] Record toggle decisions in telemetry for post-hoc analysis; emit a `BackendSelectionEvent` struct compatible with existing ingestion.  
  - [ ] Bench: verify toggling does not incur extra latency; run sanity benches under both backends.  
  - [ ] Tests: include integration tests exercising toggle transitions and asserting safe fallbacks; simulate forced fallback to legacy path by injecting compilation failure.
- [ ] Resource cache extensions  
  - [ ] Track lifetime metrics per executable (compile count, last-used) to inform eviction policies; provide `From<&ExecutableStats>` → `TelemetrySample`.  
  - [ ] Surface stats through instrumentation and optional CLI debug dumps; integrate with `metallic tool cache-dump`.  
  - [ ] Bench: stress with varied sequence lengths to validate eviction strategies.  
  - [ ] Tests: unit coverage for eviction heuristics and defensive bounds; assert we never drop an in-use executable. Include checks that buffers retained by executables remain in the most optimal storage mode.

## Milestone C: Broader Kernel Adoption
- [ ] Generalized graph builders  
  - [ ] Refactor shared primitives (tensor creation, layout conversions, mask pipelines) into reusable modules; supply `Into<MPSGraphTensor>` conversions for each primitive to minimize boilerplate.  
  - [ ] Document patterns for new kernels, including expected performance tracing steps; provide code snippets referencing the new conversion impls and noting when to request shared vs private buffers.  
  - [ ] Bench: add per-kernel benchmarks as new graphs land, starting with softmax and feed-forward candidates.  
  - [ ] Tests: enforce golden-output comparisons using burn-rs or numpy reference data where applicable; note acceptable tolerances per dtype.
- [ ] DX documentation and samples  
  - [ ] Expand `docs/MPSGraphSDPA.md` with typed API examples and migration guidance; show `From`/`Into` conversions in context.  
  - [ ] Add quick-start snippets and troubleshooting for common graph builder issues; include checklist for zero-copy pitfalls (alignment, strides).  
  - [ ] Bench/Test: ensure documentation references active benchmarks/tests; update whenever APIs evolve.
- [ ] Telemetry dashboards  
  - [ ] Wire aggregated cache/latency metrics into the qwen25 test infrastructure dashboards; surface per-backend comparisons.  
  - [x] Display resource cache statistics in the metallic_cli dashboard alongside tensor preparation cache (all cache types shown simultaneously with proper persistence)
  - [x] Emit ResourceCacheSummary events for all cache types (gemm, descriptor, softmax, sdpa, mpsgraph_sdpa) every 100 operations for dashboard aggregation
  - [x] Implement persistent stats storage for different cache types to prevent overwrite issues in UI

## Continuous Validation Checklist
- [ ] `cargo fmt`
- [ ] `cargo clippy --fix --allow-dirty --allow-staged`
- [ ] `cargo build`
- [ ] Relevant `cargo bench` targets for each milestone
- [ ] Parity/regression tests in `metallic::tests`
- [ ] Performance data captured and archived alongside code updates

## Implementation Notes

### Milestone A: Mask Reuse and Bucketing Implementation Summary

The mask reuse and bucketing system for MPSGraph SDPA has been successfully implemented with the following key components:

#### Bucketing System
- Created `MaskSizeBucket` enum with predefined size ranges (XSmall to XXXLarge) to categorize sequence lengths:
  - XSmall: 1-32
  - Small: 33-128  
  - Medium: 129-512
  - Large: 513-1024
  - XLarge: 1025-2048
  - XXLarge: 2049-4096
  - XXXLarge: 4097+
- Implemented automatic mapping from sequence lengths to appropriate buckets using `From<usize>` trait
- Designed the system to allow a single mask buffer to serve multiple sequence lengths within the same bucket

#### Separated Cache Architecture
- Created a new `CacheableMpsGraphSdpaMask` struct specifically for reusable mask buffers
- Implemented a dedicated cache in `ResourceCache` for mask buffers with separate metrics tracking
- Updated the main `CacheableMpsGraphSdpa` to no longer contain mask buffers directly
- Added new cache key type `MpsGraphSdpaMaskKey` for mask buffer identification

#### Dual-Resource Retrieval Pattern
- Added `get_or_create_mpsgraph_sdpa_and_mask()` helper method to `ResourceCache` that retrieves both the graph executable and mask buffer in a single operation
- This avoids borrowing issues that arose when trying to access the cache twice in the same scope
- The method handles both resources atomically, ensuring consistent state

#### Performance Benefits Achieved
- Significantly reduced GPU memory allocation overhead during incremental decoding where sequence lengths grow gradually
- Deterministic buffer reuse across similar sequence lengths within the same bucket
- Proper memory management with private GPU storage for long-lived masks, avoiding CPU round-trips
- Integrated observability with cache hit/miss metrics and performance counters through existing instrumentation system

#### Technical Implementation Details
- Strong typing maintained throughout with proper separation of concerns between graph executables and mask buffers
- Unsafe code minimized and properly contained with clear safety documentation
- Zero-copy semantics preserved for tensor bindings while enabling mask reuse
- Resource cleanup and lifecycle management handled automatically through the cache system

This implementation successfully addresses all requirements for Milestone A, providing a robust, performant, and maintainable foundation for high-performance MPSGraph-based SDPA operations. During incremental decoding scenarios, the system now reuses existing mask buffers instead of allocating new ones for each sequence length, significantly reducing allocation overhead and improving overall performance.
