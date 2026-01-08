# FoundryV2: Performance & Architecture Roadmap

This document outlines the current performance state of FoundryV2, findings from recent optimizations, and a roadmap for future architectural improvements.

## 1. Performance Findings

### Batching & Throughput
We observed that increasing `BATCH_SIZE` in the autoregressive decode loop significantly improves throughput by amortizing the CPU overhead of command buffer submission and synchronization.

**Benchmark Results (Qwen2.5-0.5B-Instruct, FP16):**

| Batch Size | Decode TPS | Notes |
|------------|------------|-------|
| 16         | ~81        | Initial baseline after batching implementation. |
| 32         | ~101       | **Context Parity (~105 tps)** reached. Sweet spot for latency/throughput balance. |
| 64         | ~191       | **Exceeds Context significantly.** High throughput, but longer "hiccup" every 64 tokens. |
| 128        | >1000*     | *Measurement artifact:* The entire 100-token generation finished in a single batch, resulting in near-instant decode time measurement (0.008s). In practice, this means minimal CPU overhead, but very high latency per user-visible token update. |

**Conclusion:**
- `BATCH_SIZE = 64` appears to be the optimal default for maximizing raw throughput on Apple Silicon for this model size.
- The system is now **CPU-bound** on submission overhead at lower batch sizes.
- Moving to **191 TPS** places FoundryV2 well ahead of the legacy Context engine.

### Prompt Processing Latency (GEMV vs. GEMM)
While Decode TPS is high, we observed that **Prompt Processing (Prefill)** is significantly slower than the legacy Context engine.

**Comparison:**
- **Context Engine:** ~0.45s for prompt processing.
- **Foundry Engine:** ~1.37s for prompt processing (3x slower).

**Root Cause:**
Foundry currently uses **GEMV (General Matrix-Vector Multiply)** kernels for all operations, including the prefill phase.
- **Decode Phase:** Processing 1 token at a time (vector * matrix). GEMV is optimal here.
- **Prefill Phase:** Processing N tokens at a time (matrix * matrix). GEMV is inefficient here because it treats the input as N separate vectors, missing out on matrix-matrix multiplication optimizations (tiling, reuse of weight data loaded into registers).

**Mitigation:**
We currently chunk the prefill into batches (e.g., 512 tokens) to avoid GPU timeouts, but this does not solve the fundamental arithmetic inefficiency of using GEMV for matrix multiplication.

## 2. Future Architecture Roadmap

To further push performance and architectural cleanliness, we propose the following upgrades:

### A. GEMM Kernel for Prefill
To resolve the slow prompt processing speed, we must implement a **GEMM (General Matrix-Matrix Multiply)** kernel.
- **Proposal:** Implement `GemmStep` for the prefill phase.
- **Mechanism:**
    - Detect when `seq_len > 1` (prefill) and dispatch a GEMM kernel instead of GEMV.
    - Use Metal's `simdgroup_matrix` or standard tiled threadgroup memory techniques to optimize for `M x K * K x N` multiplication.
- **Benefit:** Should bring prompt processing speed to parity with or exceed the Context engine (which uses optimized GEMM via `accelerate` or similar libs).

### B. AMX Kernel Upgrades (Component Steps)
Apple Silicon's AMX (Apple Matrix Coprocessor) offers significant acceleration for matrix multiplication beyond standard SIMD.
- **Proposal:** Implement `GemvAmxStep` and `GemvSimdgroupStep` as drop-in replacements for `GemvV2Step`.
- **Mechanism:** Use Metal's `simdgroup_matrix` (MMIT) types to leverage AMX hardware.
- **Benefit:** Higher arithmetic intensity support, freeing up ALU pipes for other operations.

### B. Indirect Command Buffers (ICB)
Currently, we loop on the CPU to encode commands for every token, even within a batch.
- **Proposal:** Use Metal Indirect Command Buffers (ICB) to encode the entire autoregressive loop (or large chunks of it) onto the GPU.
- **Mechanism:**
    1.  Create an ICB containing the `Forward` -> `Sample` -> `Copy` sequence.
    2.  Dispatch a "Driver Kernel" that executes this ICB in a loop on the GPU until EOS or max tokens.
- **Benefit:** Zero CPU overhead during generation. The CPU just waits for the final signal. This would allow `BATCH_SIZE` to be effectively infinite (limited only by VRAM/latency requirements).

### C. Advanced Kernel Fusion
Further fusion can reduce memory bandwidth pressure:
1.  **RoPE + QKV Projection:** Fuse Rotary Embedding directly into the QKV Gemv output write stage.
2.  **Gate+Up+Down MLP Fusion:** For small models, fusing the entire MLP block into a single kernel (or fewer kernels) might be possible if register pressure allows.
3.  **Logits + Sampling:** Fuse the final norm, logits projection, and sampling into a single tail kernel to avoid roundtrips.

### D. Graph Capture (Metal Performance Shaders Graph)
Investigate wrapping the entire model definition in `MPSGraph` or a custom Graph Capture mechanism that records the Metal commands once and replays them, avoiding Rust-side encoding overhead entirely.
