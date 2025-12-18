# Metallic Performance Optimization - Quick Start Guide

**Goal:** Reach **105 tok/s (FP16)** and **160 tok/s (Q8)** on M3 Pro.
**Current:** ~70.5 tok/s (FP16) | ~84 tok/s (Q8).

This guide is designed to get a new developer up to speed on the Metallic kernel architecture and the "Race to 105/160".

---

## 📂 Key File Locations

| Component | Path | Description |
| :--- | :--- | :--- |
| **FP16 Kernels** | `crates/metallic/src/kernels/matmul_gemv/kernel/dense.metal` | Contains `run_simd_f16_gemv` (The optimized kernel). |
| **Q8 Kernels** | `crates/metallic/src/kernels/matmul_gemv/kernel/quant.metal` | Contains `run_simd_q8_gemv` (The optimized quantized kernel). |
| **Logic & Headers** | `crates/metallic/src/kernels/matmul_gemv/kernel/gemv_common.h` | Shared SIMD-reduction logic and headers. |
| **Dispatch (Rust)** | `crates/metallic/src/kernels/matmul_gemv/base.rs` | Handles Metal pipeline creation and grid sizing (128 threads/TG). |
| **Dispatch (Metal)** | `crates/metallic/src/kernels/matmul_gemv/kernel/launcher.metal` | Switch-statement that routes `LoaderMode` to specific kernels. |
| **Benchmarks** | `tools/run_throughput.sh` | Main script to measure tok/s. |

## 🧠 Core Concepts & Learnings

### 1. The "SIMD-Parallel" Architecture
We moved away from "Thread-per-Column" (Legislacy) to **"Warp-per-Column"** (Modern).
- **Old Way:** 1 Thread loops `K` times. (Latency bound, low bandwidth).
- **New Way:** 32 Threads (1 Warp) collaborate on `K`. They load vector chunks, accumulate partially, and reduce via `simd_shuffle`.

### 2. Optimization Techniques Used
- **Vectorized Loads:** Always use `float4` (128-bit) loads. Reinterpret as `half` or `uchar` vectors. This is critical for M3 bandwidth.
- **Loop Unrolling:** Unroll inner loops 2x or 4x to hide global memory latency.
- **Reduction Hoisting:** **Never** put `simd_shuffle` inside the inner loop. Accumulate locally, reduce once at the end.

## 🚀 How to Run Benchmarks

**1. Standard Throughput Test:**
```bash
./tools/run_throughput.sh
```
*Look for `TPS (Total)` and `TPS (Decode)` output.*

## 🔮 Next Steps (Roadmap to 105/160)

The current kernels are efficient, The remaining gap is likely **Overhead** and **Lack of Fusion**.

### 1. Kernel Fusion (Start Here)
Fuse `RMSNorm` into the `GEMV` kernel input.
- **Why:** Removes `Read Input -> Write Norm -> Read Norm -> GEMV` round-trip.
- **Where:** Modify `dense.metal`. Add `RMSNorm` logic to the scalar read phase.

### 2. Reduce Dispatch Overhead
We currently launch ~100 kernels sequentially per token.
- **Task:** Implement **Graph Capture** (Metal Indirect Command Buffers or MPSGraph) to allow the GPU to replay the graph without CPU intervention.
- **Evidence:** See `PERFORMANCE_REPORT.md` -> analysis of CPU-side gaps.

### 3. Occupancy Tuning
Experiment with `THREADGROUP_SIZE` (currently 128) in `base.rs`. Try 256 or 512 to see if it hides latency better.

## Testing Performance
Always request that someone with a M3 Pro performs the run_throughput.sh script, and processes the results with the analyze_tmpfiles.sh to aggregate the txt files.

---
*Reference `PERFORMANCE_REPORT.md` for detailed analysis.*


