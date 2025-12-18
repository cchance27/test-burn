# Inference Framework Performance Evaluation

## Executive Summary

**Current Status (M3 Pro):**
- **FP16 Decode:** **~70.5 tok/s** (Improved from ~60 tok/s baseline).
    - *Competitor Target:* 105 tok/s.
    - *Gap:* ~33%.
- **Q8 Decode:** **~84.0 tok/s** (Improved from ~67 tok/s baseline).
    - *Competitor Target:* 160 tok/s.
    - *Gap:* ~47%.

**Conclusion:**
We have successfully moved from legacy "Thread-per-Column" kernels to optimized "SIMD-Parallel" kernels (Vectorized, Unrolled, Warp-per-Column). This yielded solid gains (+17% for FP16, +25% for Q8).

However, we are still significantly slower than state-of-the-art implementations on the same hardware. This indicates that while our **compute/bandwidth efficiency** is improving, we are hitting secondary bottlenecks, likely:
1.  **Kernel Launch Overhead**: Dispatching thousands of small kernels (Matmul, RMSNorm, RoPE, etc.) serially from CPU.
2.  **Lack of Fusion**: Competitors likely fuse `RMSNorm + Matmul` or `Matmul + Residual + Activation` to reduce memory round-trips.
3.  **Occupancy Limits**: Our 128-thread/group setup might not fully hide latency compared to larger threadgroups or different grid strategies.

---

## Detailed Analysis

### 1. Improvements Delivered (Phase 1 & 2)

#### FP16 Optimization (`gemv_f16_dense`)
- **Strategy**: Replaced legacy scalar kernel with `run_simd_f16_gemv`.
- **Techniques**:
    - **Vectorization**: 128-bit loads (`float4`) fetching 8 `half` values per op.
    - **Unrolling**: 2x Unroll (Stride 512) to hide global memory latency.
    - **Reduction Hoisting**: Moved `simd_shuffle` out of the inner loop (huge instruction saving).
- **Result**: **70.53 tok/s**.

#### Q8 Optimization (`gemv_q8_canonical`)
- **Strategy**: Replaced legacy kernel with `run_simd_q8_gemv`.
- **Techniques**:
    - **Block-Parallelism**: Processing Q8 blocks (32 weights) utilizing `uchar4` and `dot` products.
    - **Unrolling**: 4x Unroll for maximum throughput.
- **Result**: **84.02 tok/s**.

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
**Theory**: Q8 at 84 tok/s vs 160 tok/s implies we are processing at half the potential speed.
- Competitors might use `M=32` or `M=64` tile processing even for `M=1` decoding (fetching more batches or speculative decoding).
- Or they use specific `simd_permute` tricks to utilize the M3's specific ALU pipelines better.
**Target**: Deep dive into M3 ISA optimization / Assembly analysis.

## Next Steps

1.  **Profiling**: Use "Metal System Trace" (Instruments) to measure exact "Gap" time between kernel executions (CPU Overhead).
2.  **Experiment**: Implement a `FusedRMSNormGemv` kernel as a proof-of-concept.
3.  **Experiment**: Investigate `MPSGraph` for the Attention mechanism (SDPA).
