# Inference Framework Performance Evaluation

## Performance History (Latest First)

### Refactored Helpers & Pointer Optimizations
* **Change Summary**:
    * Refactored `dense.metal` and `quant.metal` to use shared `gemv_simd_impl.metal` template.
    * **Optimization**: Pre-calculated thread offsets in `init` to reduce ALU ops in the inner loop (SimdGemvPolicyF16 and SimdGemvPolicyQ8).
    * **Optimization**: Enabled "Fast Path" loop in template to remove bounds checking for bulk processing (Hot path optimization).
* **Results (M3 Pro)**:
    * **FP16 Decode**: **69.14 tok/s**
    * **Q8 Decode**: **92.76 tok/s** (+8.7 tok/s)

### Phase 1 Completion (SIMD Kernels)
* **Change Summary**:
    * Replaced legacy scalar kernels with Vectorized/Unrolled SIMD kernels.
    * Implemented `run_simd_f16_gemv` and `run_simd_q8_gemv`.
* **Results (M3 Pro)**:
    * **FP16 Decode**: ~70.5 tok/s
    * **Q8 Decode**: ~84.0 tok/s

### Baseline
* **Results (M3 Pro)**:
    * **FP16 Decode**: ~60.0 tok/s
    * **Q8 Decode**: ~67.0 tok/s

---

## The "Race to 105/160" (Phase 3 Roadmap)

To close the remaining gap, we must move beyond single-kernel optimization and look at the **graph level**.

### 1. Kernel Fusion (High Impact)
**Theory**: Every kernel launch reads inputs from RAM and writes outputs to RAM.
- Current: `Input -> RMSNorm -> RAM -> Matmul -> RAM`.
- Fused: `Input -> RMSNorm -> Regs -> Matmul -> RAM`.
**Target**: Fuse `RMSNorm` into the start of the `GEMV` kernel (Fused Add-RMSNorm-Gemv).
**Potential Gain**: +15-20%.

### 2. Dispatch Overhead Reduction (High Impact)
**Theory**: We launch kernels individually using CPU encodings (`compute_encoder.dispatch(...)`).
- Overhead can be 5-10µs per launch. A single token generation involves ~100 launches (Layers * Ops).
- 100 * 10µs = 1ms overhead per token alone.
- Competitors use `MTLHeap` or **MPSGraph** / **Metal Command Buffer Reuse** to pre-record the graph and replay it with zero CPU cost.
**Target**: Implement `Graph Capture` or `MPSGraph` backend for the model execution loop.
**Potential Gain**: +10-20% (Crucial for small batch inference).

### 3. Occupancy & Grid Tuning (Medium Impact)
**Theory**: Our current `128 threads per group` with `4 cols per group` might utilize the GPU well, but maybe `256` or `512` threads (common in Llama.cpp/MLX) would provide better latency hiding.
**Target**: Auto-tune `THREADGROUP_SIZE` and `COLS_PER_THREAD`.

### 4. Advanced Q8 Kernels
**Theory**: Q8 at 92 tok/s vs 160 tok/s implies we are processing at ~60% potential speed.
- Competitors might use `M=32` or `M=64` tile processing even for `M=1` decoding (fetching more batches or speculative decoding).
- Or they use specific `simd_permute` tricks to utilize the M3's specific ALU pipelines better.
**Target**: Deep dive into M3 ISA optimization / Assembly analysis.

## Next Steps

1.  **Phase 2**: Implement `gemv_fused.sources` to formalize kernel composition and begin Fusion work.
2.  **Profiling**: Measure CPU dispatch overhead.
