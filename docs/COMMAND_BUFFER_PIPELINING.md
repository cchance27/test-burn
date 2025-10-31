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
   - `finalize_active_command_buffer_if_latency` moves the current buffer into the pipeline. If latency mode is enabled we immediately flush all in-flight work to preserve today’s profiling semantics.
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
1. Add the pipeline module and wire it into the context layer.
2. Update GPU profiling utilities to route commit/wait bookkeeping through the pipeline.
3. Refresh documentation (`TENSOR_SYNC.md`) to describe the new pipelining behavior and responsibilities.
4. Ensure `GpuProfiler` attachment continues to work for newly created buffers.
5. Provide clear follow-up instructions for on-device validation (profiling, perf counters, stress tests).

## Validation Strategy
- `cargo +nightly fmt` locally to keep formatting consistent.
- Defer `cargo clippy` / `cargo build` / runtime validation to an Apple Silicon host; request that the team runs decode benchmarks (`analyze_jsonl.py`, standard generation loop) to confirm throughput gains.
- Once merged, enable Metal instrumented runs to verify simultaneous CPU/GPU command buffers and inspect `GpuOpCompleted` attribution for accuracy.
