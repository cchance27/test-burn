# KV Cache Layout Refactor Plan

## Objective
Stabilize autoregressive throughput by eliminating the per-token `permute` and full-history `repeat_kv_heads` copies during decode. We now keep the canonical cache in `[seq, batch_heads, head_dim]` order while maintaining a head-major repeated cache (`[batch_heads_full, seq, head_dim]`) that is updated incrementally each step. Attention consumes the repeated cache directly, so the amount of data touched per token stays proportional to the current sequence length instead of requiring an ever-growing transpose + repeat.

## Constraints & References
- Canonical cache allocation and writes still assume `[seq, batch_heads, head_dim]` ordering and use blit copies per decode step.【F:src/Metallic/context.rs†L385-L612】
- Autoregressive attention now slices a head-major repeated cache (`[batch*n_heads, seq, head_dim]`) instead of reshaping the canonical cache, avoiding the materialized permute and full-history repeat.【F:src/Metallic/models/qwen25/mod.rs†L384-L520】
- Tensor commands must keep data on-device to avoid synchronization stalls per the Metal orchestration notes.【F:TENSOR_SYNC.md†L3-L25】

## Delivery Checklist
1. **Layout Design**
   - Keep the canonical cache stored as `[seq, batch_heads, head_dim]` for write locality and document the new head-major repeated cache that mirrors it for attention.【F:src/Metallic/context.rs†L385-L459】
   - Audit cache consumers (telemetry, tests, attention) to ensure they read from the repeated cache rather than materializing a temporary transpose.【F:src/Metallic/models/qwen25/mod.rs†L438-L520】【F:src/Metallic/models/qwen25/qwen25_tests.rs†L245-L327】

2. **Allocator & Writer Updates**
   - `Context::alloc_kv_cache` allocates both the canonical cache and an optional repeated cache when GQA is enabled, zeroing all buffers from the persistent pool.【F:src/Metallic/context.rs†L385-L459】
   - `Context::write_kv_step` still performs a blit into the canonical cache and now launches `RepeatKvHeadsStepOp` to update the repeated cache incrementally, avoiding a full-history copy.【F:src/Metallic/context.rs†L460-L620】
   - Extend shape/stride assertions for both caches so misuse is caught immediately, and document the incremental update contract.

3. **Attention Path Refactor**
   - Provide `gather_repeated_cache_history` to build `[batch*n_heads, steps, head_dim]` views directly from the repeated cache without additional copies.【F:src/Metallic/models/qwen25/mod.rs†L566-L588】
   - Add `RepeatKvHeadsStepOp` so each decode iteration only repeats the freshly written timestep instead of re-emitting the entire history.【F:src/Metallic/kernels/repeat_kv_heads/mod.rs†L118-L255】【F:src/Metallic/kernels/repeat_kv_heads/kernel.metal†L34-L65】
   - Retain the legacy `repeat_kv_heads` path for contexts without GQA so existing tests continue to validate the kernel logic.

4. **Tests & Validation**
   - Add GPU tests that validate both the canonical cache slice and the repeated cache slice helpers to ensure zero-copy views expose the correct elements.【F:src/Metallic/models/qwen25/qwen25_tests.rs†L245-L327】
   - Extend the repeat kernel test suite with coverage for the incremental step kernel so we detect regressions in the new path.【F:src/Metallic/kernels/repeat_kv_heads/repeat_kv_heads_test.rs†L56-L103】
   - Run `cargo fmt`, `cargo clippy --release`, and `cargo test --release` on Apple Silicon. Because the sandbox cannot execute Metal workloads, request a teammate to run the suite and share the profiler diff once the permute call is removed.

5. **Follow-up Cleanup**
   - Remove any now-unused helper methods that were only needed for the permute path.
   - File a follow-up if you notice any remaining stride materializations elsewhere so we can iterate towards general tensor view support later.
