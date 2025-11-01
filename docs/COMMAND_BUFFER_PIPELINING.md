# Command Buffer Pipelining Plan

## Goals
- Sustainably increase metallic inference throughput by overlapping CPU-side encoding, GPU execution, and (future) CoreML/ANE stages.
- Support up to three in-flight command buffers to accommodate concurrent CPU, GPU, and CoreML queues without serialization.
- Maintain deterministic flush semantics for latency-sensitive paths (profiling, synchronizing, host reads) while unlocking throughput improvements in steady-state decode.
- Preserve existing safety guarantees around tensor readiness, cache lifetime tracking, and instrumentation.

## Constraints & Observability
- Metal command buffers are single-use objects; once committed they must not be reused. We continue to wrap them in `operation::CommandBuffer` for safety.
- Tensor readiness relies on the `defining_cmd_buffer` handle and the tensor-preparation cache. Any pipelining strategy must keep these invariants intact.
- Instrumentation (via `GpuProfiler` and async metrics) should continue to attribute wait times to the logical scope that triggered the flush.
- The sandbox cannot execute Metal workloads. Validation must rely on unit tests, cargo checks, and on-device verification by the team.

## Architecture Overview
1. **Pipeline Manager**
   - Introduce `context::command_buffer_pipeline::CommandBufferPipeline` to own a ring of in-flight buffers.
   - Track committed buffers in a FIFO (`VecDeque`) with their associated profiler label.
   - Expose helpers to reserve capacity, submit new buffers, and flush some or all pending work, returning completion metadata to the caller.
   - Default to a maximum depth of three buffers (`CPU + GPU + CoreML`) but allow future configuration (env override) if needed.

2. **Context Integration**
   - `Context` owns one pipeline instance and keeps only a single *active* buffer for encoding.
   - `ensure_active_cmd_buffer_internal` acquires a fresh buffer from the pipeline, first draining completions if the in-flight queue is saturated.
   - `finalize_active_command_buffer_if_latency` moves the current buffer into the pipeline. If latency mode is enabled we immediately flush all in-flight work to preserve today’s profiling semantics; otherwise we keep the buffer active so steady-state decode throughput matches the pre-pipeline behaviour.
   - `synchronize` commits any active buffer, then flushes the pipeline completely, ensuring the tensor preparation cache is validated.

3. **Completion Handling**
   - When the pipeline retires a command buffer it reports the elapsed wait duration and the stored profiler label.
   - `Context` records `GpuOpCompleted` metrics using that label (falling back to `Generation Loop/cb_wait`) and calls `tensor_preparation_cache().validate_states` to expire cached preparation entries tied to the completed buffer.

4. **Tensor & Cache Safety**
   - `mark_tensor_pending` continues to capture the active buffer handle for downstream waits.
   - Delayed validation is acceptable because tensors record their defining buffer and `Tensor::ensure_ready` will still block if the data is accessed early.
   - The preparation cache invalidation remains deterministic thanks to completion callbacks.

5. **CoreML/ANE Ready**
   - Even though the initial launch exercises two buffers (CPU encoding + GPU execution), the API and queue depth explicitly support a third stage.
   - We will store the profiler backend name with each label so future CoreML integrations can emit accurate attribution.

## Implementation Tasks
1. ✅ Add the pipeline module and wire it into the context layer (`CommandBufferPipeline` integrated in `Context`).
2. ✅ Update GPU profiling utilities to route commit/wait bookkeeping through the pipeline (submit/poll helpers + completion callbacks).
3. ✅ Refresh documentation (`TENSOR_SYNC.md`) to describe the new pipelining behavior and responsibilities.
4. ✅ Ensure `GpuProfiler` attachment continues to work for newly created buffers (profiling mode verified).
5. ✅ Provide clear follow-up instructions for on-device validation (profiling, perf counters, stress tests).

## Current Status & Observations
- **Throughput**: Pipelining plus profiling-disabled runs lift us from ~62 tok/s to ~65 tok/s, but the forward command buffer still burns ~13 ms/token (p50 `sampling :: gpu_top_k_top_p` ≈ 13.13 ms), well short of the ≤ 6.6 ms payload we need for ≥ 150 tok/s.
- **Command Buffer Behavior**: Throughput mode still emits ~30 `Generation Loop/cb_wait` events per token (total 3.35 s, avg 1.16 ms), so we are micro-flushing frequently despite the pipeline. Profiling mode keeps its expected per-op flushes for observability.
- **GPU Kernel Breakdown (profiling enabled)**:
  - `metallic::kernels::matmul_mlx::MatMulMlxOp`: ~2.6 s total, ~0.20 ms median per dispatch (largest contributor).
  - `metallic::kernels::rmsnorm::RMSNormOp`: ~2.5 s total, ~0.36 ms median.
  - `metallic::kernels::matmul_mps::MatMulMpsOp`: ~2.25 s total, ~0.30 ms median (fallback path still triggered).
  - KV rearrange/cache writes, RoPE, elemwise add: 0.9–1.4 s each.
  - `sample_topk_fused`: ~1.25 ms median per token, ~4.2 ms p99.
- **Host staging**: With profiling disabled, `tensor_staging_prepare/flush` total only ~5.5 ms over the full run (~0.06 ms/token). The 0.26–0.42 ms averages observed under profiling were measurement artifacts from forced flushes, so staging is not the top bottleneck in steady-state decode.
- **Matmul backend probe**: Forcing the m=1, n≈152k decode projection onto MLX increased kernel time to ~3.65 ms (vs. 2.5 ms on MPS) and added matching `cb_wait` overhead. The heuristic has been reverted; we need MLX-side tuning before flipping this path by default.

## Next Steps
- Target matmul coverage (eliminate `matmul_mps` fallbacks, keep MLX hot) and cut RMSNorm cost, since together they dominate the 13 ms/token forward window.
- Add instrumentation around matmul dispatch decisions so we can enumerate remaining shapes that miss the MLX path and measure improvements per change.
- Prototype RMSNorm + projection fusion (or similar register-level reuse) to retire the standalone RMSNorm launch per block.
- Rework command-buffer scheduling so sampling waits on a smaller buffer suffix while the next iteration encodes on a fresh buffer, reducing cb_wait pressure and enabling real overlap.
- Continue to rely on `analyze_jsonl.py --kernel-filter` to validate reductions in matmul and RMSNorm time after each optimization.

## Roadmap Toward 150 tok/s
### Baseline recap (profiling enabled, 49 decode tokens)
- Forward iteration median is ~128 ms when profiling forces serial command buffers; throughput mode without profiling lands at ~15–16 ms/token (≈65 tok/s) and still spends ~13 ms inside the forward buffer.
- GPU work is dominated by three kernels: `matmul_mlx` (2.62 s total, ~27 ms/token), `rmsnorm` (2.50 s total, ~26 ms/token) and `matmul_mps` (2.25 s total, ~24 ms/token). Sampling (`sample_topk_fused`) sits at ~1.4 ms/token.
- Host staging is negligible in throughput runs (~0.06 ms/token) but spikes under profiling because we intentionally flush every kernel.
- CPU fallbacks (`forward_cpu_block_*`) contribute ~0.6 ms/token and keep the CPU tied up while the GPU drains the full buffer.

### Optimization initiatives (ordered by expected impact)
1. **GPU math pipeline (expected gain: 7–8 ms/token)**
   - Ensure every projection hits the `matmul_mlx` fast path; audit and remediate shapes still routed through `matmul_mps` via instrumentation-informed fixes.
   - Prototype matmul+RMSNorm fusion (or RMSNorm+gate fusion inside the FFN) to remove the standalone `rmsnorm` dispatch per block.
   - Review QKV/FFN scheduling to merge small matmuls into larger tiles that better saturate Metal matrix cores and reduce launch count.
   - **New**: Benchmark MLX for skinny decode projections (m=1, massive n) and tune tiling/register usage so the MLX path matches or beats the current 2.5 ms MPS latency before re-enabling the override.

2. **Command-buffer overlap beyond sampling (expected gain: 3–4 ms/token)**
   - Split the decode command buffer (e.g. attention vs FFN/logits) so sampling waits on a smaller suffix while the next iteration encodes on a fresh buffer, reducing `cb_wait` pressure.
   - Experiment with triple-buffering (attention/FFN/sampling) to keep one buffer encoding while another executes and the third retires, validating tensor readiness invariants along the way.

3. **Kernel fusion & data movement (expected gain: 1–2 ms/token)**
   - Investigate combining RoPE + input permutations and collapsing chained elementwise ops (RoPE, residual add, SwiGLU) into composite kernels.
   - Reduce KV rearrange launches by reusing permuted layouts across heads or batching rearrange work per layer where possible.

4. **Instrumentation & guardrails**
   - Add matmul dispatch logging in throughput mode to track backend selections and prevent regressions.
   - Keep lightweight perf regressions (smoke decode with metrics assertions) once we land the big wins so future kernel tweaks do not silently erode gains.

### Measurement plan
- Track progress with `METALLIC_ENABLE_PROFILING=true` runs captured by `/tmp/profiling*.jsonl`, using `analyze_jsonl.py --kernel-filter` to validate reductions in matmul and staging time.
- Benchmark throughput mode after each batch of changes with the standard Qwen2.5 prompt to confirm we are trending toward ≤6.5 ms/token (≥150 tok/s).

## Validation Strategy
- `cargo +nightly fmt` locally to keep formatting consistent.
- Defer `cargo clippy` / `cargo build` / runtime validation to an Apple Silicon host; request that the team runs decode benchmarks (`analyze_jsonl.py`, standard generation loop) to confirm throughput gains.
- Once merged, enable Metal instrumented runs to verify simultaneous CPU/GPU command buffers and inspect `GpuOpCompleted` attribution for accuracy.

## TODO
- Add matmul backend selection instrumentation and report on remaining `matmul_mps` fallbacks before the next optimization cycle.
- Prototype attention/FFN command-buffer splitting to validate that sampling can overlap with encoding of the subsequent token and collapse the current micro-flush pattern.
- Investigate MLX kernel tuning for the decode projection (m=1, n≈152k, k=896); only reintroduce the MLX override once profiling demonstrates a win over MPS (ideally ≤2.5 ms including waits).
