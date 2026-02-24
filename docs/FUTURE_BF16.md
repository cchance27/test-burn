# BF16 Rollout Plan (Post Runtime-DType Sprint)

## Status
F16/F32 runtime dtype work is complete for this sprint.  
This document is now the implementation plan for BF16 support.

## Goals
- Add BF16 support with no silent fallback.
- Preserve current F16 performance/latency as the top constraint.
- Keep kernel DX clean via compile-time aliases and runtime helpers.
- Avoid kernel duplication and avoid runtime branching in hot loops.

## Non-Goals (This Plan)
- No broad algorithm rewrites.
- No head-dim/shape strategy redesign.
- No relaxation of fail-fast policy behavior.

## Design Contract
1. Compile-time specialization only in hot paths (`#if`, templates, aliases), not runtime `if`.
2. Keep `runtime_types.metal` as the single helper surface for storage/compute/accum helpers.
3. Keep explicit indexed vs contiguous store helper contract.
4. Mixed-policy unsupported combos remain hard errors unless explicitly implemented.
5. F16 throughput must not regress beyond benchmark noise.

## Current Foundation (Already Landed)
- Runtime type aliases and helper layer centralized in `dtypes/runtime_types.metal`.
- Fast-lane aliases added: `FastScalarT`, `FastVec2T`, `FastVec4T`.
- Key hot kernels moved from hardcoded half internals to alias-backed types:
  - GEMV (`dot.metal`, `vectorized_stage.metal`)
  - QKV (`qkv_project.metal`)
  - SwiGLU (`swiglu.metal`, `ffn_stages.metal`)
- Flash correctness/perf regressions fixed and helper contract split.

## Phase Plan

## Phase 1: BF16 Type Plumbing
- Add BF16 fast-lane mapping defines in runtime compile path:
  - `METALLIC_DTYPE_FAST_SCALAR`
  - `METALLIC_DTYPE_FAST_VEC2`
  - `METALLIC_DTYPE_FAST_VEC4`
- Add/verify BF16-aware helper conversions in `runtime_types.metal`:
  - scalar load/store
  - vec2/vec4 load/store helpers
  - pack/unpack helpers as needed for threadgroup/shared paths.
- Add strict env/contract validation:
  - reject invalid compute/accum/storage combos at compile-time boundary.

Exit gate:
- Build clean.
- Existing F16 tests pass unchanged.
- BF16 config compiles for targeted kernels.

## Phase 2: Kernel Enablement (No Perf Tuning Yet)
- Enable BF16 storage paths for dense kernels in this order:
  1. RMSNorm / RoPE / Softmax
  2. GEMV / QKV / SwiGLU
  3. FlashAttention decode/prefill/split-k
  4. GEMM/MMA
- Keep unsupported combinations fail-fast with explicit errors.

Exit gate:
- BF16 parity tests green for each family before moving to next.
- No gibberish/instability in end-to-end inference smoke runs.

## Phase 3: Parity Matrix
- Add exact tests (single test invocation style) for:
  - F16 baseline parity unchanged.
  - BF16 vs F16 tolerance per kernel family.
  - Storage/compute/accum contract validation.
  - Cache specialization correctness (dtype hash/path safety).
- Add model-level mixed-run checks:
  - dense:f16 baseline
  - dense:f32 baseline
  - dense:bf16 candidate

Exit gate:
- All parity suites pass with approved tolerances.
- No cross-dtype pipeline cache collisions.

## Phase 4: Performance Hardening
- Benchmark q4/q8/f16 baseline vs BF16 candidate.
- Profile top GPU ops and only optimize hot regressions.
- Prioritize:
  - vector load/store shape
  - conversion chain removal in hot loops
  - index-width preservation (`32` in hot path unless required).
- Keep regression checks after each micro-change.

Exit gate:
- F16 performance remains in current baseline range.
- BF16 path performance is accepted for targeted models/devices.

## Phase 5: Docs + Policy Surface
- Update docs table for supported storage/compute/accum combos.
- Add clear examples for policy variant usage.
- Document fail-fast rules for unsupported BF16 combos.

Exit gate:
- Single authoritative dtype-support matrix published.

## Kernel Family Checklist (BF16)
- [ ] runtime_types helper layer
- [ ] rmsnorm
- [ ] rope
- [ ] softmax / sdpa materialized
- [ ] gemv
- [ ] qkv
- [ ] swiglu / fused ffn
- [ ] flashattention decode
- [ ] flashattention prefill / split-k
- [ ] gemm / mma
- [ ] sampling (dtype contract validation where relevant)

## Testing Checklist
- [ ] exact kernel parity tests for each family (F16 vs BF16)
- [ ] exact runtime-helper regression tests (indexed/contiguous helpers)
- [ ] exact pipeline-cache dtype specialization tests
- [ ] model-level smoke generation tests (no corruption)
- [ ] throughput benchmarks per target model

## Benchmark Protocol
1. `cargo build -q --message-format=short`
2. run one exact test at a time (`--exact`)
3. run throughput scripts (q4, q8, fp16 models)
4. compare top-op timings and decode/prefill TPS
5. reject changes that regress F16 hot path materially

## Definition of Done
1. BF16 storage/compute/accum support is implemented for all targeted kernel families.
2. Unsupported combinations fail fast with explicit diagnostics.
3. F16 performance remains stable relative to current recovered baseline.
4. BF16 parity and model-level smoke tests pass.
5. Docs reflect final supported matrix and policy controls.
