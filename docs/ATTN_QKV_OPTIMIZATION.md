# attn_qkv_proj Optimization Notes

This document summarizes actionable strategies to reduce the latency and memory footprint of the `attn_qkv_proj` phase recorded by our instrumentation during autoregressive decoding.

## Current Hot Path

During `forward_step` each transformer block currently issues three independent GEMMs followed by bias adds to produce Q, K, and V projections before head packing and RoPE:

* `ctx.matmul` is called once per projection with `transpose_b = true`.
* A `BroadcastElemwiseAddOp` adds the bias tensor for each projection.
* Each tensor is subsequently rearranged into per-head layouts for attention.

Because these operations run back-to-back they occupy the same latency band that our metrics report as **attn_qkv_proj**.

## Optimization Opportunities

### 1. Fuse the QKV projection into a single dispatch

* Concatenate the Q/K/V weights along the output dimension and pre-bake the matching biases so that we can issue **one** `ctx.matmul` and **one** fused bias add (or a custom kernel that performs both). This removes two command-buffer submissions and two bias kernels per block.
* Packing the fused weights upfront also reduces repeated reads from Metal buffers, aligning with the synchronization model that keeps work inside a single command buffer until the host touches results.【F:TENSOR_SYNC.md†L3-L25】【F:src/metallic/models/qwen25/mod.rs†L379-L394】
* If we introduce a dedicated Metal kernel for `linear_bias_fused` we can record only one `LatencyEvent::block_phase` for the entire projection, improving profiling clarity while avoiding CPU fallbacks that would break batching.【F:TENSOR_SYNC.md†L13-L25】【F:INSTRUMENTATION.md†L5-L28】

### 2. Store weights in the optimal layout

* The matmul currently requests a transpose of every projection weight (`transpose_b = true`). Persisting the weights in column-major/row-major form that matches Metal's preferred orientation eliminates the runtime transpose branch and allows us to call `ctx.matmul` without the transpose flag.【F:src/metallic/models/qwen25/mod.rs†L385-L392】
* Re-exporting checkpoints with pre-transposed weights trades a one-time conversion for per-token savings, honoring our performance-first rule set.【F:PROJECT_RULES.md†L3-L23】

### 3. Batch bias and reshape work on GPU

* `BroadcastElemwiseAddOp` launches separate compute passes that can stall if they trigger host-visible tensors. Extending the fused linear kernel to output `[batch, seq, 3 * head_dim]` and then slicing/views for Q/K/V keeps all work on-GPU and leverages the implicit synchronization pipeline.【F:TENSOR_SYNC.md†L3-L25】【F:src/metallic/models/qwen25/mod.rs†L384-L404】
* After fusion, adjust `KvRearrangeOp` (or replace it with a kernel that directly emits `[heads, head_dim]` tiles) so that we avoid redundant tensor reshapes or CPU-driven loops flagged as remaining sync pain points.【F:TENSOR_SYNC.md†L13-L24】【F:src/metallic/models/qwen25/mod.rs†L402-L429】

### 4. Instrument and validate each change

* Use the existing latency collectors to confirm the **attn_qkv_proj** row shrinks after each optimization. Capturing before/after snapshots in Ratatui or JSONL logs keeps regressions visible.【F:INSTRUMENTATION.md†L5-L58】
* Pair measurements with targeted micro-benchmarks (e.g., a criterion bench around the fused projection) so we can iterate without waiting for full-model runs. Ensure any new kernel obeys the implicit synchronization contract to prevent unexpected host waits.【F:TENSOR_SYNC.md†L3-L30】【F:INSTRUMENTATION.md†L5-L58】

## Next Steps

1. Prototype a fused `linear_bias_fused` operation in Metal, including weight packing tooling when loading GGUF weights.
2. Update the Qwen25 block to consume the fused tensor and slice Q/K/V views before RoPE.
3. Validate with the metrics dashboard and ask a teammate to benchmark on-device since we cannot execute Metal workloads inside CI.

The fused projection now ships as `Context::fused_qkv_projection`, which emits separate Q/K/V tensors via the Metal kernel in `kernels/fused_qkv`. Weight packing happens during GGUF load so runtime matmuls no longer require transposed operands.【F:src/metallic/context.rs†L205-L299】【F:src/metallic/models/qwen25/loading.rs†L77-L358】

Following these steps should turn **attn_qkv_proj** from the second-slowest phase into a smaller slice of the block latency budget while maintaining clean layering and synchronization safety.
