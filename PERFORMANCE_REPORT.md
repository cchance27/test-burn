# Inference Framework Performance Evaluation

## Performance History (Latest First)

### KV Cache & Throughput Measurement Optimizations
* **Change Summary**:
    * **Deterministic Benchmarking**: Added `--seed` flag for reproducible token generation.
    * **Silent Mode**: Introduced `output-format=none` to eliminate terminal I/O and GPU rendering contention.
    * **KV Cache Scaling**: Removed redundant $O(n)$ full-history repetition in Qwen 2.5 `forward_step`, ensuring $O(1)$ scaling during decode.
    * **Statistical Reporting**: Enhanced `run_throughput.sh` with 10-run averaging and Min/Avg/Max reporting.
* **Results (M3 Pro, MAX_TOKENS=256, 10-run avg)**:
    * **FP16 Decode**: **99.51 tok/s** (Max: 100.74)
    * **Q8 Decode**: **149.68 tok/s** (Max: 150.33)
* **Impact**: Throughput measurements are now stable and free from terminal/scaling artifacts. FP16 performance remains top-tier while ensuring deterministic results.

### Unified SIMD Backend & Legacy Cleanup
* **Change Summary**:
    * **Unified Architecture**: Ported legacy "Dense" FP16 kernels to use the optimized `run_simd_gemv_template` used by Q8.
    * **Legacy Removal**: Removed manual "Thread-per-Column" dense loops; all kernels now use "Warp-per-Column" SIMD logic.
    * **Correctness**: Validated against reference implementation.
* **Results (M3 Pro)**:
    * **FP16 Decode**: **~100 tok/s** (+~7% vs ~93 tok/s)
    * **Q8 Decode**: **~150 tok/s** (+~7% vs ~140 tok/s)
* **Impact**: Codebase significantly simplified. Performance gap between FP16 and Q8 narrowed. Foundation laid for further fusion.

### RMSNorm-GEMV Fusion (Q8) + GEMV Variants
* **Change Summary**:
    * Added RMSNorm-fused GEMV for Q8 paths (QKV + SwiGLU), reducing extra read/write traffic.
    * Added `METALLIC_GEMV_COLS_PER_TG=2|4|8` variants for GEMV tuning.
    * **Important:** Dense FP16 fused GEMV was disabled due to weight layout mismatch - now resolved in Layout Unification above.
* **Results (M3 Pro, MAX_TOKENS=256, run_throughput.sh)**:
    * **FP16 Decode**: **~63–65 tok/s** (regression vs prior ~70 tok/s)
    * **Q8 Decode**: **~130–133 tok/s** (+~35 tok/s vs prior)
* **Notes**:
    * Profiling runs (`run_throughput_w_prof.sh`) are **much slower** and should not be compared to non-prof throughput numbers.

### Fused SwiGLU & Occupancy Tuning
* **Change Summary**:
    * Implemented `gemv_q8_swiglu_f16`, fusing Gate, Up, and SiLU into a single kernel launch.
    * Replaced 3 discrete kernel calls (Gate, Up, Act) with 1 fused call per MLP layer.
    * Verified 128-thread occupancy remains optimal.
* **Results (M3 Pro)**:
    * **FP16 Decode**: **71.62 tok/s** (+1 tok/s)
    * **Q8 Decode**: **97.75 tok/s** (+1.3 tok/s)

### "Fast Path" & Pointer Optimizations
* **Change Summary**:
    * **Build Fixed**: Resolved header guard and redefinition errors.
    * **Optimization**: `SimdGemvPolicyQ8` now absorbs thread offsets into base pointers and uses unchecked `load_x_fast` for bulk processing.
* **Results (M3 Pro)**:
    * **FP16 Decode**: **70.68 tok/s** (Stable)
    * **Q8 Decode**: **96.48 tok/s** (+29.5 tok/s from baseline!) 
        * *Note*: Improvement from ~84 to ~96 tok/s solely from pointer/loop optimizations.

### Refactored Helpers & Pointer Optimizations (Initial Attempt)
* **Change Summary**:
    * Refactored `dense.metal` and `quant.metal` to use shared `gemv_simd_impl.metal`.
    * Pre-calculated thread offsets.
* **Results (M3 Pro)**:
    * **FP16 Decode**: 69.14 tok/s
    * **Q8 Decode**: 92.76 tok/s

### Phase 1 Completion (SIMD Kernels)
* **Change Summary**:
    * Replaced legacy scalar kernels with Vectorized/Unrolled SIMD kernels.
* **Results (M3 Pro)**:
    * **FP16 Decode**: ~70.5 tok/s
    * **Q8 Decode**: ~84.0 tok/s

### Baseline
* **Results (M3 Pro)**:
    * **FP16 Decode**: ~60.0 tok/s
    * **Q8 Decode**: ~67.0 tok/s

---

## The "Race to 105/160" (Phase 3 Roadmap)

To close the remaining gap (~96 -> 160), we must move beyond single-kernel optimization.

### 0. ~~Unify Dense/Q8 GEMV Layouts~~ ✅ RESOLVED
**Status**: Completed in Layout Unification update.
**Summary**: All dense weights now transposed during loading via `copy_weight_transposed_into_fused`, producing `[In, Out]` layout. Dense path uses `transpose_right=false` matching Q8.
**Impact**: FP16 can now use the same SIMD helpers and fused kernels as Q8.

### 1. WMMA/AMX and Layout Optimizations
**Theory**: WMMA/AMX can provide significant performance improvements for GEMV and related operations.
**Target**: Implement WMMA/AMX for GEMV and related operations.

### 2. Unified Backend Architecture (Completed)
**Status**: **Completed**.
**Summary**: Unified FP16 Dense and Q8 backends to use a single `SimdGemvPolicy` template.
**Impact**: FP16 performance improved to ~100 tok/s, Q8 to ~150 tok/s. Legacy code removed.

### 2. Kernel Fusion 
**Theory**: Every kernel launch reads inputs from RAM and writes outputs to RAM.
**Target**: Fuse `RMSNorm` into `GEMV` or `SwiGLU` (Gate+Up+Act) into single kernel.
**Status**:
*   **Q8**: Fused RMSNorm+QKV and SwiGLU live.
*   **FP16**: Fused RMSNorm+GEMV and SwiGLU+GEMV are now implemented for the Canonical path.

### 3. Canonical Dispatch Completeness
**Status**: **Mostly Complete**.
*   **Prefill (M>1)**: Dispatcher now supports `DenseCanonical` for M>1 via blocked-row-major decoding in `gemm`.
*   **Selection**: `TransformerBlock` now conditionally loads *only* canonical weights when enabled, enforcing the optimized path at the data level.

### 3. Dispatch Overhead Reduction (High Impact)
**Theory**: CPU dispatch overhead is significant (~10µs/launch).
**Target**: Implement `MPSGraph` or `Metal Graph Capture`.

### 4. Occupancy & Grid Tuning (Medium Impact)
**Target**: Auto-tune `THREADGROUP_SIZE` and `COLS_PER_THREAD`.

### 5. Advanced Q8 Kernels
**Theory**: Q8 at 96 tok/s vs 160 tok/s implies we are processing at ~60% potential speed.
**Target**: Deep dive into M3 ISA optimization.
