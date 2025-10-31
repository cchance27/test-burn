# Sample TopK/TopP GPU Profiling Guide

This note captures the agreed baseline workflow for investigating the `sample_topk_topp` kernels and gathering actionable GPU timing data. Follow these steps whenever we need to re-profile or verify optimisations.

## 1. Enable Instrumentation Outputs

1. Pick a writable path for metric dumps, e.g. `/tmp/sample_topk_metrics.jsonl`.
2. Export the required environment variables before running any binaries:

```bash
export METALLIC_ENABLE_PROFILING=1
export METALLIC_METRICS_JSONL_PATH=/tmp/sample_topk_metrics.jsonl
# Optional: quiet console spam if desired
export METALLIC_METRICS_CONSOLE=0
```

3. Run `cargo test -p metallic --features metal -- --nocapture sample_topk_topp::benchmark_cpu_vs_gpu_sampling` to refresh the latency baseline. The updated test now:
   - Emits structured duration summaries (avg/p50/p95/stddev) for CPU and GPU phases.
   - Records kernel dispatch metadata and per-op completions into the JSONL file via `metallic_instrumentation`.
   - Verifies that the partial and merge kernels are producing profiler events so we catch regressions early.

The resulting JSONL feed contains both `GpuKernelDispatched` (records threadgroup geometry) and `GpuOpCompleted` events with full op paths, which makes it easy to slice the data in notebooks or lightweight dashboards.

## 2. Capture Metrics for Offline Analysis

After the test completes you can ingest `/tmp/sample_topk_metrics.jsonl` with the tools of your choice (pandas, polars, etc.). Recommended schema filters:

- `event.type == "GpuKernelDispatched"` and `kernel_name` starting with `sample_topk` gives per-phase threadgroup and occupancy hints.
- `event.type == "GpuOpCompleted"` scoped under `sample_topk_topp_benchmark/…` yields CPU-side encode durations plus Metal GPU timings.

Keep an archive of these raw runs in the performance history folder so we can compare deltas when iterating.

## 3. Xcode GPU Trace Workflow

We rely on Xcode's GPU Frame Debugger for deep kernel inspection. The simplified flow:

1. Build a small harness that calls the sampler once (the benchmark test is fine — run with `--nocapture` so prints appear). Launch it under Xcode: **Product → Scheme → Edit Scheme → Run → Info** and set the executable to `cargo`. Supply arguments `test -p metallic -- --nocapture sample_topk_topp::benchmark_cpu_vs_gpu_sampling`.
2. In the scheme diagnostics tab enable **Metal API Validation** and **GPU Frame Capture**.
3. Run the scheme once; when the benchmark reaches the GPU section Xcode will offer a capture icon in the debug bar. Click it right before the sampling loop starts.
4. After the capture completes, Xcode opens the Metal frame debugger. Filter the command buffer list for `sample_topk_partials` or `sample_topk_merge_and_sample` — you will see threadgroup sizing, grid dimensions, and hardware counters (thread occupancy, memory bandwidth) for each dispatch.
5. Save the `.gputrace` file into the `perf/` directory with a timestamp (e.g. `perf/sample_topk_topp_2024-03-02.gputrace`). This lets us diff statistics between revisions.

### Quick sanity checks inside Xcode

- Verify that both kernels show the expected number of threadgroups (matches JSONL). Large divergences usually mean our scheduling heuristic fell back to a slow path.
- Inspect the **Encoder Statistics** pane: pay special attention to `Threadgroup Utilization`, `SIMD Group Efficiency`, and `L2/L1 Read Transactions`. These highlight whether upcoming SIMD-group refactors are targeting the right bottleneck.
- Use the **Counters** timeline to ensure that RNG/softmax work is not serialized; unusually tall spikes suggest we stayed single-threaded somewhere.

## 4. Repeating the Baseline

Whenever we land kernel changes:

1. Re-run the benchmark test with profiling enabled (step 1) and stash the JSONL output.
2. Capture a fresh `.gputrace` session following step 3.
3. Drop the summary JSON printed by the test into the performance tracking sheet together with key metadata: git commit, GPU model, macOS version.

Keeping both the structured metrics and the Xcode trace ensures we have cross-validated evidence before and after each optimisation pass, helping us avoid regressions while we chase the performance targets.
