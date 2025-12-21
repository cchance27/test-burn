# Metallic Performance Optimization - Quick Start Guide

**Goal:** Reach **105 tok/s (FP16)** and **160 tok/s (Q8)** on M3 Pro.
**Current (FAST Architecture):** **105.47 tok/s (FP16 decode)** | **158.52 tok/s (Q8 decode)** using `MAX_TOKENS=50` on M3 Pro.
**Latest Status:** Batched & Strided architecture enabled. KV cache overhead minimized. Statistical benchmarking integrated.

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

## Audited and Analyzed benchmarks
Latest non-prof runs are tracked in:
- `metallic-throughput-fp16.txt`
- `metallic-throughput-q8.txt`
Profiling runs (much slower, for attribution only):
- `prof-metallic-throughput-fp16.txt`
- `prof-metallic-throughput-q8.txt`


## 🛠️ Kernel Build System

Metallic uses a custom build system to manage Metal kernels, enabling both modular development (includes) and optimized release packaging (precompiled binaries).

### 1. The `.sources` Manifest
Instead of writing monolithic Metal files, we use `kernel.sources` manifests to stitch together reusable components.
- **File:** `crates/metallic/src/kernels/.../kernel.sources`
- **Format:** A simple list of relative paths to other `.metal` files (e.g., `../common/helpers.metal`).
- **Build Process:** `build.rs` reads this manifest and generates a temporary `{kernel_name}_includes.metal` file containing `#include` directives for every listed file. This allows multiple kernels to share common logic without code duplication.

### 2. `build.rs` Logic
The `build.rs` script performs the following steps:
1.  **Discovery:** Recursively scans `crates/metallic/src/kernels` for `kernel.metal` (standalone) or `kernel.sources` (composite).
2.  **Generation:** For `.sources`, it generates the include file.
3.  **Compilation:** Invokes `xcrun metal` to compile `.metal` to `.air`, and then `xcrun metallib` to link `.air` to `.metallib`.
4.  **Output:** All compiled binaries are placed in the `OUT_DIR` environment directory.

### 3. The `kernel_lib!` Macro
To support both rapid development and optimized releases, we use the `kernel_lib!` macro (defined in `crates/metallic/src/macros.rs`) to load kernels:

| Mode | Feature Flag | Behavior |
| :--- | :--- | :--- |
| **Development** | `src_kernels` (or debug) | Loads the kernel source code as a string using `include_str!`. Compiles at runtime (JIT). Allows hot-reloading/fast iteration. |
| **Release** | `built_kernels` (default) | Loads the precompiled `.metallib` binary using `include_bytes!`. Fast startup, no runtime compilation overhead. |

**Example Usage in `mod.rs`:**
```rust
KernelLibrary::MatmulGemv => kernel_lib!("matmul_gemv"),
```
This automatically switches between source and binary loading based on your Cargo features.

## 🧠 Core Concepts & Learnings

### 1. The "SIMD-Parallel" & Batched Architecture
We moved away from "Thread-per-Column" (Legacy) to **"Warp-per-Column"** (Modern) and now support **Multi-Batch Execution**.
- **Batching:** Kernels now utilize the Metal grid `depth` to process multiple batches in parallel, with offsets managed via strided parameters.
- **Strided Access:** All GEMV kernels now use a unified `GemvParams` struct containing `stride_x`, `stride_y`, `stride_a`, `stride_w`, and `stride_scale` for flexible memory layouts.
- **Layouts:** Q8 uses transposing (`transpose_right=false`). FP16 uses **Canonical Blocked Layout** (K-major blocks) with `cols8` tiling for maximum M3 bandwidth utilization.

### 2. Optimization Techniques Used
- **Vectorized Loads:** Always use `float4` (128-bit) loads. Reinterpret as `half` or `uchar` vectors. This is critical for M3 bandwidth.
- **Loop Unrolling:** Unroll inner loops 2x or 4x to hide global memory latency.
- **Reduction Hoisting:** **Never** put `simd_shuffle` inside the inner loop. Accumulate locally, reduce once at the end.

## 🚀 How to Run Benchmarks

**1. Standard Throughput Test:**
```bash
./tools/run_throughput.sh
```
*Reports Statistical Min/Avg/Max for `TPS Total` and `TPS Decode` across 5 iterations (configurable via `ITERATIONS` env).*

**2. Silent Mode (Benchmark Only):**
```bash
cargo run --release -- [model] [prompt] --output-format none --seed 42
```
*Suppresses token printing to eliminate GPU contention/IO overhead.*

**2. Variant Tuning (GEMV cols per threadgroup):**
```bash
METALLIC_GEMV_COLS_PER_TG=2 ./tools/run_throughput.sh
METALLIC_GEMV_COLS_PER_TG=4 ./tools/run_throughput.sh
METALLIC_GEMV_COLS_PER_TG=8 ./tools/run_throughput.sh
```

**Note:** `run_throughput_w_prof.sh` is intentionally slow and should only be used for per-kernel attribution.

## 🔮 Ideas for Next Steps (Roadmap to 105/160)

The current kernels are efficient, The remaining gap is likely **Overhead** and **Lack of Fusion**.

### 0. ~~Unify Dense/Q8 Layouts~~ ✅ RESOLVED
Dense weights now loaded into optimized `CanonicalF16Tensor` format.
- **Result:** FP16 now uses the same SIMD helpers and fused kernel architecture as Q8. Legacy layouts removed.

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
We now support `METALLIC_GEMV_COLS_PER_TG=2|4|8` (maps to threadgroup widths 64/128/256) for GEMV variants.
Use `cargo bench -q --bench gemv_variant_bench -- --warm-up-time 1 --measurement-time 3` and compare runs with different env values.

## Known Issues & Parity Gaps
- **Output Projection (Prefill)**: Still uses dense `[N,K]` + `transpose_b=true` when `m>1` - however, bit-perfect parity is now achieved.
- **Fused Kernels (Decode)**:
    - **RMSNorm+QKV**: Fully implemented and verified for FP16 Canonical.
    - **RMSNorm+SwiGLU**: Fully implemented and verified for FP16 Canonical.
- **Prefill GEMM**: Optimization needed; FP16 prefill is currently ~1.5x slower than Q8.
- **Dispatcher**: Manual selection remains in model code for the absolute fastest path, though canonical support is broadening, This is critical to finish

## Testing Performance
Always request that someone with a M3 Pro performs the run_throughput.sh and run_throughput_w_prof.sh scripts, and processes the results with the analyze_tmpfiles.sh to aggregate the txt files. For your review with full performance mode and the prof_ files that include the kernel materialization (much slower but allows us to see individual step/kernel comparisons and % of time spent in each step/kernel)

metallic-throughput-fp16.txt = FP16 throughput
metallic-throughput-q8.txt = Q8 throughput
prof-metallic-throughput-fp16.txt = FP16 throughput with kernel materialization
prof-metallic-throughput-q8.txt = Q8 throughput with kernel materialization

---
*Reference `PERFORMANCE_REPORT.md` for detailed analysis.*
