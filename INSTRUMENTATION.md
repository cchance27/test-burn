# Instrumentation Overview

The instrumentation layer is split into two cooperating pieces:

* `src/metallic/instrumentation.rs` exposes lightweight collectors that the
  Metal context installs while kernels run.  The collectors record structured
  latency (`LatencyEvent`) and memory (`MemoryEvent`) measurements for the
  forward pass, individual transformer blocks, and any labelled block phases.
  Model code (for example `src/metallic/models/qwen25/mod.rs`) emits events
  around attention/MLP calls so the collectors can capture fine‑grained timing
  and allocation deltas.
* `src/metallic/metrics.rs` converts the collector snapshots into the data used
  by the Ratatui dashboard.  It maintains rolling statistics for latency,
  aggregates per-scope memory usage, builds the static model weight tree, and
  optionally persists metrics to disk.

## Latency tracking

The generation loop attaches a latency collector before each forward step.  The
collector records:

* The overall embedding, forward, output, and sampling durations
* Per-block totals for every transformer layer
* Optional, model-defined phases (e.g. SDPA, attention projections, MLP stages)

`metrics::build_latency_rows` turns the rolling statistics into the hierarchical
rows shown in the sidebar so individual phases can be expanded underneath each
block.

### KV cache lifecycle instrumentation

KV caching now maintains a single attention-ready buffer per transformer layer
with layout `[batch * n_heads, seq, head_dim]`. `Context::write_kv_step` writes
new timesteps directly into this layout, avoiding the previous canonical
mirroring pass. Incremental generation reuses the unified cache directly, so
the `kv_repeat` latency phase continues to capture the light append work rather
than an expensive replay. The dedicated KV cache pool still has an 8 GB safety
limit, but generation sizes each allocation to the active prompt window
(`prompt_tokens + max_new_tokens`, clamped to the model's maximum). The memory
collector reports the per-layer footprint using this per-run capacity so the UI
mirrors the smaller reservation instead of the legacy full-sequence sizing.

### Softmax backend benchmarking

Softmax normalization now supports two execution paths: the existing
`sdpa_fused_softmax` compute kernel and a path that records
`MPSMatrixSoftMax` through the shared `ResourceCache`. The desired backend is
selected via the `METALLIC_SOFTMAX_BACKEND` environment variable:

* `auto` (default) keeps the compute pipeline for all legacy call sites but
  allows attention code paths that opt in to request the MPS variant when it is
  legal (i.e. no causal masking or query offsets).
* `kernel`, `compute`, or `pipeline` force the fused compute shader.
* `mps` or `metal` force the MPS softmax wherever the caller allows it.

During generation, `Context` records per-dispatch softmax timings and the
metrics layer aggregates them into two new top-level latency rows, **Softmax
(Kernel)** and **Softmax (MPS)**. These show the most recent and rolling average
latencies for each backend so you can compare them live in Ratatui or via the
JSONL logs. Use the new rows to validate the faster option on-device before
changing the default for production builds.

For offline validation, run the dedicated criterion benchmark to compare both
paths outside of the generation loop:

```
cargo bench --bench softmax_backend_benchmark
```

The benchmark reports separate `kernel` and `mps` measurements using identical
tensor shapes so you can determine which backend is faster on a given piece of
hardware.

## Memory tracking

The memory collector captures pool usage, KV cache growth, and per-phase deltas.
`metrics::build_memory_rows` merges these snapshots with the static weight
breakdown to display reserved pool capacity, KV cache usage, and model weight
contributions.  Host memory is sampled through `ProcessMemoryTracker`, allowing
us to attribute baseline and unattributed host usage alongside GPU allocations.

## JSONL logging

Metrics logging is opt-in.  Set the `METRICS_LOG_ENABLED` environment variable to
`1`, `true`, `yes`, or `on` to emit timestamped JSONL files.  The default logging
interval is five seconds; override it with `METRICS_LOG_INTERVAL_SECS` (in
seconds).  Enabling logging creates `<timestamp>-latency.jsonl` and
`<timestamp>-memory.jsonl` files in the working directory, each containing an
array of the rows currently shown in the UI.

## UI integration

`metrics::MetricsLoggers` streams the formatted rows to disk, while the
Ratatui UI receives the same data through `AppEvent::LatencyUpdate` and
`AppEvent::MemoryUpdate`.  This separation keeps `generation.rs` focused on the
inference workflow and routes all presentation-specific formatting through the
metrics module.
