# Instrumentation Overview

The instrumentation layer is split into two cooperating pieces:

* `src/Metallic/instrumentation.rs` exposes lightweight collectors that the
  Metal context installs while kernels run.  The collectors record structured
  latency (`LatencyEvent`) and memory (`MemoryEvent`) measurements for the
  forward pass, individual transformer blocks, and any labelled block phases.
  Model code (for example `src/Metallic/models/qwen25/mod.rs`) emits events
  around attention/MLP calls so the collectors can capture fine‑grained timing
  and allocation deltas.
* `src/Metallic/metrics.rs` converts the collector snapshots into the data used
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
