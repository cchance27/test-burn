# MLX Kernel Dev Startup — Quick Guide

This is the minimum you need to add, run, and compare a new M=1 NT kernel variant in the enhanced harness — without reading all the Swift files.

## TL;DR

- Edit or add Metal kernels in `tools/metal_probes/matmul/*.metal`.
  - The build script compiles every `.metal` in that folder into one `.metallib` per file.
  - Give your kernels stable `[[host_name("...")]]` symbols (see examples below).
- Register variants in `tools/metal_probes/matmul/variants_enhanced.json`.
  - `library` = metallib base name (file name without `.metal`)
  - `functionName`/`kernel_name_override` = the `[[host_name]]` string
  - `enabled: true` to include in sweeps
- Run:

```bash
rm -rf tools/metal_probes/.cache tools/metal_probes/.build/matmul
METAL_RUN=1 ./tools/metal_probes/run_matmul_probes_enhanced.sh
```

The harness prints per‑shape GPU/CPU times and max errors.

## Where Things Live

- Kernels: `tools/metal_probes/matmul/*.metal`
- Variant manifest: `tools/metal_probes/matmul/variants_enhanced.json`
- Harness script: `tools/metal_probes/run_matmul_probes_enhanced.sh`
- Swift harness (reference only):
  - `tools/metal_probes/main_harness.swift`
  - `tools/metal_probes/variant_manager.swift`
  - `tools/metal_probes/generic_kernel_runner.swift`
  - `tools/metal_probes/base_backend_runner.swift`
  - `tools/metal_probes/matmul_types.swift`
- Shapes: `tools/metal_probes/MATMUL_QWEN25_SIZES.md`
- Notes & Plans: `tools/metal_probes/MLX_KERNEL_NOTES.md`, `tools/metal_probes/MLX_KERNEL_PLANS.md`

## Add a New Kernel — 5 Steps

1) Create or edit a `.metal` file in `matmul/`

```metal
// Example: tools/metal_probes/matmul/m1_dot_product_v5.metal
#include <metal_stdlib>
using namespace metal;

[[kernel]]
void your_kernel_impl(...);

// Bind a stable name the harness can call:
template [[host_name("m1_dot_product_v5_nt_bn128_col_vec4_tgread_bk64_tg128")]] [[kernel]]
void your_kernel_impl(...);
```

Tips:
- Put `[[kernel, max_total_threads_per_threadgroup(TG)]]` on the kernel definitions, not on explicit instantiations (MSL restriction).
- Avoid `alignas` on `threadgroup` arrays; use plain declarations.
- Use scalar head/tail around vector loads for alignment (e.g., align K to 4 for `half4`).
- Keep barriers minimal but correct around double‑buffered TG tiles.

2) If creating a brand‑new backend key (e.g., `m1_optimized_v5`)

- Add enum case in `tools/metal_probes/matmul_types.swift` → `MatmulShapeSpec.Backend`.
- Add the backend to `VariantManager.backendDisplayOrder` in `tools/metal_probes/variant_manager.swift`.
- `generic_kernel_runner.swift` already supports M=1 backends; dispatch derives from name tokens.

3) Register a variant in `variants_enhanced.json`

```json
{
  "name": "nt_bn128_col_vec4_tgread_bk64_tg128",
  "library": "m1_dot_product_v5",
  "kernel_name_override": "m1_dot_product_v5_nt_bn128_col_vec4_tgread_bk64_tg128",
  "is_m1_specific": true,
  "supports": {
    "transposeA": false,
    "transposeB": true,
    "smallK": true,
    "smallMN": true,
    "batch": false,
    "bias": false,
    "accumulate": false,
    "functionName": "m1_dot_product_v5_nt_bn128_col_vec4_tgread_bk64_tg128",
    "expectedTransposeA": false,
    "expectedTransposeB": true
  },
  "enabled": true
}
```

Notes:
- `library` must match your `.metal` filename (without extension).
- `functionName`/`kernel_name_override` must match your `[[host_name("...")]]` string.
- Use the standard naming tokens (below) for free dispatch.

4) Build and run

```bash
rm -rf tools/metal_probes/.cache tools/metal_probes/.build/matmul
METAL_RUN=1 ./tools/metal_probes/run_matmul_probes_enhanced.sh
```

Useful env:
- `MATMUL_BENCH_ITERS` (default 8), `MATMUL_BENCH_WARMUP` (default 2)
- `MATMUL_BACKENDS` to restrict backends, e.g. `MATMUL_BACKENDS=m1_optimized_v4,m1_optimized_v3`
- `MATMUL_COMPARE_ALL=true|false` to compare across all enabled backends or only the spec default

5) Document outcomes

- Add findings to `MLX_KERNEL_NOTES.md` and plans to `MLX_KERNEL_PLANS.md`.
- Disable losing variants in `variants_enhanced.json` (`enabled: false`) to keep sweeps focused.

## How Dispatch Works (no heuristics here)

The harness infers launch geometry from tokens in the variant name:

- `bnXX` → columns per threadgroup (e.g., `bn128` = 128 columns/TG)
- `tgYY` → threads per threadgroup (e.g., `tg64` = 64 threads/TG)
- Defaults: `bn128`, `tg128` if no token present

Grid = `(N + bn - 1) / bn` TGs in X; 1×1 in Y×Z. Tokens match only when followed by digits (so `tgread` is not `tg`).

## Reading Results

- GPU time: `gpuStartTime/gpuEndTime` (accurate when dispatch is valid)
- CPU time: host round‑trip
- Correctness: f16 maxRel ≈ `4.8e-4` typical; `maxAbs` ≈ `O(1e-2)` scaled by data range
- “skipped/unsupported” in the summary often means capability flags filtered a spec (not a math error)

## Common Pitfalls

- Don’t put attributes (like `max_total_threads_per_threadgroup`) on explicit instantiations.
- Don’t use `alignas` on threadgroup storage (MSL forbids attributes on TG memory).
- Always add scalar head/tail when using `half4`/`float4` to handle alignment and tails.
- Keep barriers only where truly necessary around double‑buffered A tiles.
- If you see `0.0ms` GPU + huge error, verify variant name tokens and that your kernel symbol matches the manifest.

## Naming Cheatsheet

`nt_bn128_col_vec4_tgread_bk64_tg128`

- `nt` = transposeB=true (NT path)
- `bn128` = 128 columns per TG
- `col_vec4_tgread` = A‑tiling; per‑lane TG reads for A; `half4` for B
- `bk64` = K tile size used by the kernel (cosmetic, good for clarity)
- `tg128` = 128 threads per TG

## “Add New Backend” Checklist

1. Add backend enum in `matmul_types.swift`.
2. Add backend to `VariantManager.backendDisplayOrder`.
3. Define variants under that backend in `variants_enhanced.json`.
4. Provide kernels and `[[host_name]]` symbols in a `.metal` file.
5. Run the script; analyze; record decisions in NOTES/PLANS.

## Reference Commands

```bash
# One‑shot rebuild + run
rm -rf tools/metal_probes/.cache tools/metal_probes/.build/matmul
METAL_RUN=1 ./tools/metal_probes/run_matmul_probes_enhanced.sh

# Restrict backends for faster loops
MATMUL_BACKENDS=m1_optimized_v4,m1_optimized_v3 \
  METAL_RUN=1 ./tools/metal_probes/run_matmul_probes_enhanced.sh

# Tune iteration counts
MATMUL_BENCH_ITERS=6 MATMUL_BENCH_WARMUP=2 \
  METAL_RUN=1 ./tools/metal_probes/run_matmul_probes_enhanced.sh
```

## Final Notes

- The harness is intentionally heuristic‑free; selection heuristics belong in Metallic, not here.
- Keep the default suite lean — disable parity/losers in the manifest and record why in NOTES/PLANS.

