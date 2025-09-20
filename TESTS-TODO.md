# Test Coverage Roadmap for Metallic Module

This document tracks planned test coverage for the metallic module across SDPA end-to-end, Softmax kernel, and MatMul components. Each item includes scenarios and acceptance criteria to guide implementation.

## A. SDPA End-to-End Tests

- [x] Tensor helper unit tests (host-side)
  - [x] zeros/ones and zeros_like/ones_like
  - [x] element-wise ops (+, -, *, /) and fill
  - [x] get_batch and from_existing_buffer
  - Acceptance: Unit tests compile and pass on macOS targets

- [ ] Shape diversity
  - [ ] Non-square, non-power-of-two shapes (e.g., `batch=1, seq_q=7, seq_k=13, dim=5`). Compare against Burn output within tolerance.
  - [ ] Very small dimensions (e.g., `dim=1`, `seq_q=1`, `seq_k=1`) — verify results and no panics.
  - [ ] Larger odd sizes to stress reductions (e.g., `seq_k=257`, `seq_q=31`, `dim=63`). Compare against Burn within tolerance.
  - [ ] Multiple batches (e.g., `batch=3`) with varied contents (uniform shapes). Compare against Burn within tolerance.
  - Acceptance criteria: All comparisons to Burn pass; no panics; correct output shapes.

- [ ] Causality correctness
  - [ ] For `is_causal=true`, verify output does not depend on future keys/values. Method: change V (and/or K) in masked positions and confirm output remains unchanged.
  - Acceptance criteria: Output invariant to masked-position changes; differences remain within floating-point tolerance.

- [ ] Row-stochastic property
  - [ ] For random shapes and values (both `is_causal=false` and `is_causal=true`), verify each attention row sums to ~1.0 within tolerance (`rtol=1e-4`, `atol=1e-6`).
  - Acceptance criteria: All rows sum to 1.0 within tolerance; no NaNs/Infs.

- [ ] Numerical stability
  - [ ] Inputs with large magnitudes or shifts (e.g., add +1000.0 to Q·K^T logits). Compare against Burn; ensure stable results.
  - Acceptance criteria: No NaNs/Infs; close match to Burn within tolerance; correct normalization of softmax.

- [ ] Property-based tests (optional, stretch)
  - [ ] Randomized shapes within bounds (e.g., `batch<=3`, `seq_q<=65`, `seq_k<=513`, `dim<=129`) and random seeds; compare against Burn.
  - Acceptance criteria: Sampled cases pass consistently across multiple seeds; test runtime remains reasonable.

## B. Softmax Kernel Tests (Kernel-level)

- [ ] Causal and non-causal golden tests
  - [ ] Small attention matrices (e.g., `seq_q=2`, `seq_k=4`) with known values.
  - [ ] `is_causal=false`: Verify equality to CPU softmax per-row within tolerance and rows sum to ~1.
  - [ ] `is_causal=true`: Verify masked elements effectively zeroed and rows sum to ~1; match CPU masked-softmax.
  - Acceptance criteria: Exact or tight tolerance match to CPU results; no NaNs/Infs.

- [ ] Irregular sizes and reduction robustness
  - [ ] Odd/non-power-of-two `seq_k` (e.g., 13), with `seq_q` = 1..N (e.g., 7); verify row sums and CPU match.
  - [ ] Larger `seq_k` values (e.g., 257) to stress reduction schedule; verify row sums and CPU match.
  - Acceptance criteria: Row sums ~1.0 and CPU match across irregular sizes; no panics.

- [ ] Numerical extremes
  - [ ] Rows with large positive/negative values; rows with identical values; rows with one very large outlier.
  - Acceptance criteria: No NaNs/Infs; stable normalization; expected behavior vs CPU.

- [ ] Pipeline compilation/caching
  - [ ] Call `ensure_fused_softmax_pipeline` multiple times; ensure idempotence (no re-compilation side-effects; no panics).
  - Acceptance criteria: Compiles once; repeated calls are fast and safe.

- [ ] Threadgroup width compatibility
  - [ ] Validate execution using the pipeline’s native `threadExecutionWidth()` (>= 32), ensuring reduction steps remain correct.
  - [ ] For devices/reporting with smaller widths, verify we clamp appropriately and correctness is maintained.
  - Acceptance criteria: Kernel produces correct results with the chosen threadgroup size on all supported hardware.

## C. MatMul Tests (MPS Integration)

- [ ] Correctness vs CPU (small and medium shapes)
  - [ ] Small integer test: `2x3 · 3x2 → 2x2` with simple inputs; compare exact CPU matmul.
  - [ ] Asymmetric shape: `5x4 · 4x7 → 5x7` with random floats; compare against CPU within tolerance.
  - Acceptance criteria: Exact match for small integer test; tolerance match for floats.

- [ ] Transpose flags coverage
  - [ ] `transpose_right=true`: e.g., `A(2x3) · B(2x3)^T → 2x2` correctness.
  - [ ] `transpose_left=true`: e.g., `A(2x3)^T · B(2x3) → 3x3` correctness.
  - Acceptance criteria: Correctness within tolerance for both transpose paths.

- [ ] Alpha/Beta accumulation
  - [ ] Verify `result = alpha * A·B + beta * C` with non-zero `C`. Choose `alpha=0.5`, `beta=0.25` and assert expected results.
  - Acceptance criteria: Matches expected accumulations within tolerance.

- [ ] Non-zero buffer offsets and rowBytes
  - [ ] Allocate larger buffers; place matrices at non-zero byte offsets; build `MPSMatrixDescriptor` with correct `rowBytes`.
  - [ ] Validate we can read/write at offsets and results are correct.
  - Acceptance criteria: Correct results and no out-of-bounds; validates `mps_matrix_from_buffer` and descriptor use.

## D. Cross-Cutting & Regression Safeguards

- [ ] Performance smoke tests (optional)
  - [ ] Quick-run benchmarks for a few shapes to ensure no large regressions in wall time (not strict perf tests, but sanity checks).
  - Acceptance criteria: No order-of-magnitude slowdowns across commits.

- [ ] Determinism checks (where possible)
  - [ ] Validate results are deterministic across multiple runs for the same inputs (within floating point tolerance).

- [ ] Error-path tests
  - [ ] Validate error handling for invalid shapes/dimension mismatches in Tensor, MatMul descriptor creation, and SDPA.
  - Acceptance criteria: Clear error returned; no panics.

## Implementation Notes

- Use Burn reference (`sdpa_burn.rs`) as the CPU/baseline for SDPA comparisons.
- Use simple CPU implementations for softmax/masked-softmax and matmul in tests where Burn is not required.
- Use tolerances `rtol=1e-4`, `atol=1e-6` for float comparisons unless smaller is warranted by the scenario.
- Prefer smaller, fast tests; reserve large cases for targeted validation.
- When testing offsets, ensure alignment and `rowBytes` match `columns * sizeof(f32)`.

---

Owner: Metallic Module
Scope: CI-required tests where practical; others can be weekly/nightly.
Priority order:
1) Softmax kernel golden tests + SDPA shape diversity
2) MatMul transpose/alpha-beta/offset tests
3) SDPA causality invariance + row-stochastic + numerical stability
4) Property tests and performance smoke tests
