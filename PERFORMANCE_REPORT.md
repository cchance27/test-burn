# Inference Framework Performance Evaluation

## Performance History (Latest First)

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

### 1. WMMA/AMX and Layout Optimizations
**Theory**: WMMA/AMX can provide significant performance improvements for GEMV and related operations.
**Target**: Implement WMMA/AMX for GEMV and related operations.

### 2. Kernel Fusion 
**Theory**: Every kernel launch reads inputs from RAM and writes outputs to RAM.
**Target**: Fuse `RMSNorm` into `GEMV` or `SwiGLU` (Gate+Up+Act) into single kernel.

### 3. Dispatch Overhead Reduction (High Impact)
**Theory**: CPU dispatch overhead is significant (~10µs/launch).
**Target**: Implement `MPSGraph` or `Metal Graph Capture`.

### 4. Occupancy & Grid Tuning (Medium Impact)
**Target**: Auto-tune `THREADGROUP_SIZE` and `COLS_PER_THREAD`.

### 5. Advanced Q8 Kernels
**Theory**: Q8 at 96 tok/s vs 160 tok/s implies we are processing at ~60% potential speed.
**Target**: Deep dive into M3 ISA optimization.
