# Metal SIMD-Group Probe Kernels

These throwaway kernels let us validate Metal 4.0 SIMD-group intrinsics in isolation before wiring them into the real sampling kernels. A Swift harness executes the compiled metallibs to verify correctness across deterministic and randomized scenarios before we touch production code.

## Probes

- `simdgroup_reduce_max.metal` – shuffle-down reduction validated against ascending, random, and tail-masked inputs.
- `simdgroup_ballot_compact.metal` – exercises `simd_ballot`, manual mask reconstruction, prefix compaction, and edge cases (none/all active, sparse/dense random, tail-masked).
- `simdgroup_register_topk.metal` – keeps a per-lane register shortlist tested with monotonic, random, and masked value sets.

## Usage

Compile a probe with `xcrun`:

```bash
xcrun -sdk macosx metal -std=metal4.0 simdgroup_reduce_max.metal -o simdgroup_reduce_max.air
xcrun -sdk macosx metallib simdgroup_reduce_max.air -o simdgroup_reduce_max.metallib
```

Run the helper script to compile (and optionally execute) every probe in this directory:

```bash
./run_probes.sh                  # compile
# METAL_RUN=1 ./run_probes.sh    # compile + execute Swift harness
```

The script emits errors if compilation fails; with `METAL_RUN=1` it also runs `run_probes.swift`, printing PASS/FAIL summaries for every scenario.
