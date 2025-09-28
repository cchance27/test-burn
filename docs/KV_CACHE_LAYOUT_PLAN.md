# KV Cache Layout Refactor Plan

## Objective
Stabilize autoregressive throughput by eliminating the per-token `permute` over the full KV history. Rather than relayout the cache storage, we will teach the attention path (specifically `repeat_kv_heads`) to consume the existing `[seq, batch_heads, head_dim]` layout so that `gather_cache_history` can hand back a zero-copy slice.

## Constraints & References
- Current cache allocation and writes assume `[seq, batch_heads, head_dim]` ordering and use blit copies per decode step.【F:src/Metallic/context.rs†L385-L477】
- Attention gathers the entire cache history, slices the active window, and issues a `permute([1, 0, 2])`, making cost scale with the context length.【F:src/Metallic/models/qwen25/mod.rs†L439-L565】
- Tensor commands must keep data on-device to avoid synchronization stalls per the Metal orchestration notes.【F:TENSOR_SYNC.md†L3-L25】

## Delivery Checklist
1. **Layout Design**
   - Confirm the cache remains stored as `[seq, batch_heads, head_dim]` and document that `repeat_kv_heads` now operates directly on this layout.
   - Audit other cache consumers (e.g., telemetry or unit tests) to confirm no assumptions remain about a permuted `[batch_heads, seq, head_dim]` view.

2. **Allocator & Writer Updates**
   - No allocator change is required; `Context::alloc_kv_cache` already provides contiguous `[seq, batch_heads, head_dim]` buffers.【F:src/Metallic/context.rs†L379-L468】
   - Keep `Context::write_kv_step` validation aligned with the seq-major layout and call out in documentation that no device-side permute is expected.【F:src/Metallic/context.rs†L426-L468】
   - Extend any shape/stride assertions so misuse is caught immediately, and add doc comments describing the zero-copy contract.

3. **Attention Path Refactor**
   - Replace `Qwen25::gather_cache_history` with a view builder that slices `[0..steps]` in-place, marks tensors safe for command submission, and returns the view without calling `permute`.【F:src/Metallic/models/qwen25/mod.rs†L545-L565】
   - Update `repeat_kv_heads` (Rust wrapper and Metal kernel) to interpret inputs as `[seq, batch_heads, head_dim]`, mirroring the cache layout and avoiding the materialized transpose.【F:src/Metallic/kernels/repeat_kv_heads/mod.rs†L24-L96】【F:src/Metallic/kernels/repeat_kv_heads/kernel.metal†L1-L28】

4. **Tests & Validation**
   - Update `qwen25_tests::test_gather_cache_history_gpu_path` (and related fixtures) to assert the direct `[steps, batch_heads, head_dim]` slice now returned from the cache.【F:src/Metallic/models/qwen25/qwen25_tests.rs†L245-L282】
   - Run `cargo fmt`, `cargo clippy --release`, and `cargo test --release` on Apple Silicon. Because the sandbox cannot execute Metal workloads, request a teammate to run the suite and share the profiler diff once the permute call is removed.
   - Capture before/after latency logs from the inference harness to confirm tokens/sec no longer degrades as sequence length grows.

5. **Follow-up Cleanup**
   - Remove any now-unused helper methods that were only needed for the permute path.
   - File a follow-up if you notice any remaining stride materializations elsewhere so we can iterate towards general tensor view support later.
