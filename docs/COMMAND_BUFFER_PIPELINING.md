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
- **Throughput**: With pipelining enabled and profiling disabled we improved from ~62 tok/s to ~65 tok/s. Latency per token remains ~15–16 ms, indicating GPU kernels dominate runtime.
- **Command Buffer Behavior**: In throughput mode we keep a single command buffer per iteration; sampling submits the buffer to the pipeline and overlaps CPU prep of the next token. In profiling mode every op is flushed individually, so there is no overlap (expected) but per-kernel timing is accurate.
- **GPU Kernel Breakdown (profiling enabled)**:
  - `metallic::kernels::matmul_mlx::MatMulMlxOp`: ~2.6 s total, ~0.20 ms median per dispatch (largest contributor).
  - `metallic::kernels::rmsnorm::RMSNormOp`: ~2.5 s total, ~0.36 ms median.
  - `metallic::kernels::matmul_mps::MatMulMpsOp`: ~2.25 s total, ~0.30 ms median (fallback path still triggered).
  - KV rearrange/cache writes, RoPE, elemwise add: 0.9–1.4 s each.
  - `sample_topk_fused`: ~1.25 ms median per token, ~4.2 ms p99.
- **Host staging**: `tensor_staging_prepare/flush` appear more frequently than expected (0.26–0.42 ms avg); likely triggered by CPU-side embedding / tokenizer reads.

## Next Steps
- Reduce per-token matmul time (ensure MLX path, explore fusing norms, investigate splitting attention vs FFN buffers for additional overlap).
- Audit CPU fallbacks (embedding, `forward_cpu_block_*` call sites) and minimize staging to keep tensors resident on GPU.
- Continue using `analyze_jsonl.py` with filters (e.g. `--kernel-filter rmsnorm`) to track hot kernels after each optimization.
- Explore deeper pipelining opportunities:
  - Evaluate splitting forward-pass work across multiple command buffers (attention vs FFN) to keep the GPU busy while the CPU prepares the next token.
  - Assess whether streaming/decode work can move to a dedicated thread so UI callbacks do not block the main decode loop once the sampled token has been read.
  - Investigate staging hotspots (`tensor_staging_prepare/flush`) to ensure we are not accidentally forcing synchronous host reads between kernels.

## Roadmap Toward 150 tok/s
### Baseline recap (profiling enabled, 49 decode tokens)
- Forward iteration median is ~128 ms when profiling forces serial command buffers; throughput mode without profiling lands at 15–16 ms/token (≈65 tok/s).
- GPU work is dominated by three kernels: `matmul_mlx` (2.62 s total, ~27 ms/token), `rmsnorm` (2.50 s total, ~26 ms/token) and `matmul_mps` (2.25 s total, ~24 ms/token). Sampling (`sample_topk_fused`) sits at ~1.4 ms/token.
- Host staging costs are non-trivial: `tensor_staging_prepare/flush` together burn ~949 ms over the run (~19 ms/token) and signal that we are bouncing tensors between CPU and GPU.
- CPU fallbacks (`forward_cpu_block_*`) contribute ~0.6 ms/token and keep the CPU tied up while the GPU finishes the current command buffer.

### Optimization initiatives (ordered by expected impact)
1. **GPU math pipeline (expected gain: 3–4 ms/token)**
   - Ensure every projection lands on the `matmul_mlx` fast path; audit shapes that still route through `matmul_mps` and either realign dimensions or extend the MLX backend.
   - Prototype matmul+RMSNorm fusion (or RMSNorm+gate fusion inside FFN) to remove the standalone `rmsnorm` dispatches and reduce kernel count per block.
   - Review QKV/FFN scheduling to combine small matmuls into fewer, larger tiles that better saturate the Metal matrix cores.

2. **Eliminate avoidable host staging (expected gain: 2–3 ms/token)**
   - Track down why `tensor_staging_prepare/flush` fires on every iteration; keep embedding, logits, and sampling buffers resident on GPU until the stream callback actually needs host access.
   - Push token decode and streamer callbacks onto a dedicated thread that consumes device-to-host copies asynchronously so kernel submission for the next token never waits on `as_slice()`.

3. **Command-buffer overlap beyond sampling (expected gain: 2–3 ms/token)**
   - Split the decode command buffer into at least two stages (attention vs FFN) so CPU prep of the next iteration can start after attention commits instead of waiting for the entire forward pass.
   - Experiment with triple-buffering (attention/FFN/sampling) to keep one buffer encoding while another executes and the third drains, validating tensor readiness invariants as we go.

4. **Kernel launch count & data movement (expected gain: 1–2 ms/token)**
   - Investigate opportunities to fuse RoPE + matmul input transforms and to collapse chained elementwise ops (RoPE, add, SwiGLU) into composite kernels using our existing Metal graph utilities.
   - Reduce KV rearrange invocations by reusing permuted layouts across heads or batching rearrange work per layer where possible.

5. **Longer-term bets**
   - Evaluate CoreML/ANE offload for matmul-heavy paths on supported hardware once the Metal path is lean, using the pipeline’s third slot.
   - Add lightweight perf regressions (smoke decode with metrics assertions) so future kernel changes do not silently erode gains.

### Measurement plan
- Track progress with `METALLIC_ENABLE_PROFILING=true` runs captured by `/tmp/profiling*.jsonl`, using `analyze_jsonl.py --kernel-filter` to validate reductions in matmul and staging time.
- Benchmark throughput mode after each batch of changes with the standard Qwen2.5 prompt to confirm we are trending toward ≤6.5 ms/token (≥150 tok/s).

## Validation Strategy
- `cargo +nightly fmt` locally to keep formatting consistent.
- Defer `cargo clippy` / `cargo build` / runtime validation to an Apple Silicon host; request that the team runs decode benchmarks (`analyze_jsonl.py`, standard generation loop) to confirm throughput gains.
- Once merged, enable Metal instrumented runs to verify simultaneous CPU/GPU command buffers and inspect `GpuOpCompleted` attribution for accuracy.
