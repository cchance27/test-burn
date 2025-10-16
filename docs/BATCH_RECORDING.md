# Batch Recording of GPU Operations

NOTE: We tested using this and it seemed to be less efficient due to cpu overhead implementing batched ops..

We added a batched API on the `CommandBuffer` that allows encoding multiple operations while sharing encoders.
This reduces encoder churn (creation/teardown) when recording sequences of operations.

Public API

- `CommandBuffer::record(&self, op: &dyn Operation, cache: &mut ResourceCache)`
  - Records a single operation. By default we reuse the active encoder across similar operations and only end encoders when switching between blit and compute, or at `commit()`.
- `CommandBuffer::record_batch(&self, ops: &[&dyn Operation], cache: &mut ResourceCache)`
  - Encodes multiple operations and ends the active encoder once after the batch.
  - Preserves per-operation GPU profiling because each `Operation::encode` opens its own profiler scope.

When to use batching

- Use batching when you already have a known list of operations to record together, particularly when the sequence may switch between blit and compute encoders.
- For many small operations, batching helps ensure we close encoders only once per batch, reducing overhead.

Notes on profiling

- Profiling remains per-operation. Each operation is responsible for starting and finishing a `GpuProfiler` scope in its `encode()` call.
- `record_batch` defers encoder closure until the batch completes; it does not merge or collapse profiler scopes.

Benchmarking

- We added a mixed-op benchmark `record_vs_batch_record_mixed` with N in {1, 10, 100} that alternates blit and compute operations to force encoder switches:
  - Individual: records each op separately.
  - Batched: records the same ops via `record_batch`.
- This benchmark is intended to surface encoder churn differences more clearly than single-kind workloads.

Caveats

- The default `record()` already reuses encoders within the same kind, so batching may not show improvement for homogeneous op sequences (e.g., all blit).
- Real gains appear when encoder switching is frequent or when many tiny ops are batched.
