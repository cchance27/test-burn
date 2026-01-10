# FoundryV2: Performance & Architecture Roadmap

This document outlines the current performance state of FoundryV2, findings from recent optimizations, and a roadmap for future architectural improvements.

## 1. Performance Findings

### Batching & Throughput
We batch some decode work in Foundry to reduce CPU submission overhead, but large batch sizes can distort user-visible latency and also distort metrics if we attribute "prompt processing" using time-to-first-token (TTFT).

**Key takeaway:** treat TTFT-based measurements with caution when the decode loop is explicitly batched.

### Prompt Processing Latency (GEMV vs. GEMM)
FoundryV2 supports `m>1` prefill (GEMM) and can be very fast for prefill when measured correctly. Prefill throughput can look extremely high if the prompt length is small relative to the measured interval; always compare prompt token counts and ensure we are measuring the actual prefill command buffer time (not TTFT).

## 1.1 Current Known Perf Gaps vs `context` (As of 2026-01-09)
On Qwen2.5-0.5B-Instruct FP16, FoundryV2 currently shows:
- Prefill: very high throughput with `m>1` GEMM prefill
- Decode: at/near parity with `context` (after SDPA + KV cache fixes)

The earlier decode gap was not explained by GEMM/GEMV microbench parity alone. It came from **pipeline-level** overhead: avoidable per-step uploads/waits, extra dispatches, and memory placement differences.

### Reference Numbers (Qwen2.5-coder-0.5B-instruct FP16)
Using `./tools/run_throughput.sh --fp16 --max-tokens 256` (5 iterations) on 2026-01-09:
- `context`: ~98 tok/s decode, ~86 tok/s prefill, ~58 tok/s end-to-end
- `foundry`: ~97 tok/s decode, ~520–550 tok/s prefill, ~90 tok/s end-to-end

### Why FoundryV2 Can Be Slower Even When Kernels Are “Similar”
FoundryV2’s compound kernels primarily fuse *within a single op* (stage composition inside one kernel). FoundryV2 does not (yet) have a graph-level “op fusion pass” that merges multiple JSON ops into a single dispatch.

By contrast, the `context` engine’s Qwen25 path uses a number of **macro-fused ops** (multiple logical operations in one kernel) and can route some matmul shapes through **MLX / MPS-style kernels** that are faster than our V2 path for certain decode-dominant shapes.

### Missing Macro-Fusions (Decode-Critical)
These fusions exist in `context` for Qwen25 but do not exist as FoundryV2 ops today:

1) **MatMul “addmm” fusion on GEMV path**
   - Fuses: `A*B + beta*C (+ bias)` for decode matmul (common in residual adds).
   - FoundryV2 recently reduced some of this overhead by folding residual into the GEMV write stage, but parity still requires the larger macro-fusions that cover FFN and (potentially) attention output patterns.

2) **Tail fusion (lower ROI than SDPA/FFN)**
   - Fuses: final RMSNorm + logits matmul (+ sampling).
   - FoundryV2 currently performs these as separate steps (and uses a dedicated sampling kernel).

### Implemented Macro-Fusions (2026-01-09)
1) **FFN fused path**
   - Implemented as Foundry op `FusedFfnSwiGluRmsNorm` (RMSNorm + gate/up + SwiGLU) and wired into `models/qwen25.json`.
   - Parity test: `metallic::tests::fused_ffn_swiglu_rmsnorm_parity`.

### Backend / Memory Placement Differences (Not “Fusion”, Still Major)
1) **MatMul backend mismatch**
   - Benchmarks show V2 matmul is slower than MLX for some important decode shapes (e.g. `M=1, N=896, K=4864`).
   - `context` can route those through MLX/MPS-style implementations; FoundryV2 currently does not.

2) **StorageModeShared vs private**
   - FoundryV2 currently materializes GGUF tensors into `StorageModeShared` buffers and allocates many intermediates as `StorageModeShared`.
   - `context` tends to keep hot buffers in GPU-private memory.
   - This can dominate performance even if “the compute kernel is the same”.

## 2. Future Architecture Roadmap

To further push performance and architectural cleanliness, we propose the following upgrades:

### A. GEMM Kernel for Prefill
FoundryV2 has a GEMM path for `m>1` prefill. The remaining work here is primarily **tuning** (tile config selection, packing/layout selection, and avoiding extra scratch work), not initial bring-up.

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
2.  **FFN (Decode) Macro-Fusion:** Implemented as `FusedFfnSwiGluRmsNorm`; next step is evaluating whether to include the down-projection + residual/bias in the same op (register pressure permitting).
3.  **Gate+Up+Down MLP Fusion (Future):** For small models, fusing the entire MLP block into a single kernel (or fewer kernels) might be possible if register pressure allows.
3.  **Logits + Sampling:** Fuse the final norm, logits projection, and sampling into a single tail kernel to avoid roundtrips.

### D. Graph Capture (Metal Performance Shaders Graph)
Investigate wrapping the entire model definition in `MPSGraph` or a custom Graph Capture mechanism that records the Metal commands once and replays them, avoiding Rust-side encoding overhead entirely.

## 3. Implemented: FFN Decode Fusion via CompoundKernel Composition
Implemented as of 2026-01-09: `FusedFfnSwiGluRmsNorm`.

### 3.1 Implemented Op: `FusedFfnSwiGluRmsNorm` (Decode-only, `m==1`)
**Goal:** replace 4–5 dispatches per layer with 1 dispatch in the FFN block for `m==1` decode.

**Inputs:**
- `x`: `[1, d_model]` (or `[d_model]`)
- `gamma`: `[d_model]` (RMSNorm)
- `w_gate`: `[ff_dim, d_model]` (NK row-major) or canonical (choose one and stick to it)
- `b_gate`: `[ff_dim]` (optional)
- `w_up`: `[ff_dim, d_model]`
- `b_up`: `[ff_dim]` (optional)

**Output:**
- `hidden`: `[1, ff_dim]` (SwiGLU output)

**Computation:**
1) RMSNorm(x) → x_norm
2) gate = x_norm @ w_gate^T (+ b_gate)
3) up   = x_norm @ w_up^T (+ b_up)
4) out  = swiglu(gate, up)
5) write `out`

### 3.2 Implementation Sketch (CompoundKernel)
This is feasible as a single kernel by:
- Using a warp-per-row dispatch for `N=ff_dim` outputs.
- Computing both dot-products (gate and up) in the same K loop to maximize reuse of `x_norm` loads.
- Performing a single reduction per output row, then applying bias + swiglu and writing.

This likely requires adding a new “dual dot + reduce + swiglu write” stage (or a dedicated `.metal` include) because our current `WarpReduceStage` and dot stages assume a single accumulator value.

### 3.3 Tests
Parity tests compare:
- `RmsNormV2 + MatMul(gate) + MatMul(up) + SwigluV2` vs `FusedFfnSwiGluRmsNorm`
- Cover `m==1` first, then extend to `m>1` later if desired.

### 3.4 Integration Plan
1) Add new Foundry op (Step + CompiledStep) and kernel. (done)
2) Update `models/qwen25.json` to use the fused op. (done)
3) Later: add a graph rewrite pass to auto-fuse the pattern so specs stay clean. (todo)
