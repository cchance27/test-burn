# GGML vs Metallic (Metal) — Analysis, Findings, and Migration TODO

This document summarizes where GGML’s Metal backend currently outperforms our Metallic kernels, outlines specific shortcomings and missteps in Metallic, and proposes a structured TODO plan to close the gap. It is designed to be self-contained so we don’t need to jump back to GGML frequently while planning and implementing improvements.

References in this repo:
- Metallic kernels and backends:
  - Matmul: `crates/metallic/src/kernels/matmul_mps/`, `crates/metallic/src/kernels/matmul_mlx/`, `crates/metallic/src/kernels/matmul_gemv/`, and dispatcher in `crates/metallic/src/kernels/matmul_dispatcher/`
  - SDPA: `crates/metallic/src/kernels/scaled_dot_product_attention/`, softmax: `crates/metallic/src/kernels/softmax/`
  - Kernel manager/cache: `crates/metallic/src/kernels/`
- GGML Metal backend (reference):
  - `experimental/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `experimental/ggml/src/ggml-metal/ggml-metal.cpp`
  - `experimental/ggml/src/ggml-metal/ggml-metal-common.h`

Notes about our current state (as context):
- We currently default to MPS for certain GEMM paths when MLX cannot outperform it (contrary to prior assumption that we generally prefer MPS). For small-N cases, we are NOT using a GEMV specialization because our earlier GEMV implementation underperformed.
- Performance priorities for Qwen2.5-like inference: matmul (GEMM/GEMV) and SDPA dominate runtime.
- We are not moving to Flash Attention yet; our goal is to optimize our existing primitives.


## 1) Executive summary

- Matmul: GGML dynamically selects among multiple specialized kernels (true GEMM, various mat-vec variants), tuned for shapes and device features (e.g., simdgroup matrix multiply). It uses shared memory aggressively, adjusts threadgroup sizes per kernel, and avoids extra layout transforms. Metallic now has a robust shape-aware dispatcher that selects the best variant, with MLX/MPS fallbacks. We have implemented the infrastructure for high-performance small-N GEMV family, though the actual kernel implementation remains for future work.
- SDPA: Metallic’s SDPA pipeline is a 3-stage sequence (QK matmul, softmax with optional causal mask, AV matmul). With the 5.1 improvements, we now have shape-aware matmul dispatch for QK/AV legs, parameterized softmax threadgroup sizing, and logical transpose preference for K. GGML’s attention stack (even without relying solely on Flash) demonstrates strong per-shape specialization, simdgroup reductions, tuned threadgroup configurations, and minimizes transfers/permutes.
- General: GGML caches and selects pipelines based on data type, shape buckets, and device caps; it designs kernels to minimize barriers and global memory passes. Metallic now has a more explicit specialization and dispatch layer for kernels following the completion of 5.1 quick wins.

## 2) Where GGML surpasses Metallic

2.1 Shape-aware dispatch and kernel families
- GGML switches between matmul variants based on N (number of columns of RHS), K alignment, and device caps:
  - For small N (e.g., N ∈ [1..8]): use mat-vec style kernels with tuned threadgroup sizes.
  - For larger M,N: use true GEMM with simdgroup MM where supported.
- Multiple variants for attention/softmax, selecting block vs vector kernels based on head size and batch/sequence configuration.

2.2 Threadgroup tiling and simdgroup usage
- Uses simdgroup matrix multiply (when supported) and simdgroup reduction intrinsics to minimize shared-memory barriers and synchronization overhead.
- Tunes threadgroup sizes (e.g., nxpsg, r1ptg, etc.) to match shape divisibility and hardware wavefront size.

2.3 Memory layout and transfers
- Avoids explicit permutes by leveraging strides (logical transpose) wherever possible.
- Carefully stages tiles in threadgroup memory only when beneficial; otherwise, leans on vectorized global loads.

2.4 Softmax with reductions
- Implements reductions using simdgroup operations and/or tiered reductions with fewer barriers.
- Parameterizes parallel configuration by row length (sequence length), preventing underutilization for short rows and bottlenecks for long rows.

2.5 Pipeline caching and specialization keys
- Pipelines are keyed by dtype, tile size bucket, shape class (e.g., small-N vs mm), and flags (causal, scaling).
- This lets GGML avoid runtime compilation costs and select the correct variant consistently.


## 3) Metallic shortcomings and probable missteps

3.1 Lack of a robust shape-aware dispatcher for matmul
- ~We rely primarily on MLX (and use MPS where MLX loses) but do not route small-N cases to a specialized GEMV family. Earlier GEMV underperformance led us to avoid it entirely, leaving a significant gap for inference shapes where N is small.~ **RESOLVED in 5.1**: Shape-aware dispatcher implemented with strongly-typed enums and exhaustive match logic.

3.2 Under-optimized softmax reductions
- Current softmax kernel uses a fixed 256-threadgroup reduction via shared memory. This can be:
  - Over-provisioned for short rows (wasted threads) or
  - Insufficiently parallel for very long rows (not scaling with sequence length).
- We do not use simdgroup intrinsics for reductions; this increases barrier overhead and latency. **PARTIALLY RESOLVED in 5.1**: Parameterization infrastructure added, but simdgroup reductions remain for 5.2.

3.3 Overheads from layout transforms
- Although we support transpose flags, we still sometimes materialize transposes or copies for K (in QK^T) when a logical transpose via strides would suffice, especially when not routing through MLX. **RESOLVED in 5.1**: SDPA now defaults to logical transpose preference.

3.4 Inconsistent α/β fused update usage
- In some paths we still do separate add or store operations instead of a true αAB + βC fused pipeline, causing extra memory traffic. **RESOLVED in 5.1**: Consistent α/β fused usage implemented across all matmul paths.

3.5 Kernel cache specialization not granular enough
- Cache keys don’t fully encode tile size buckets, small-N specialty, transpose flags, β!=0, etc., causing suboptimal reuse or incorrect reuse. **RESOLVED in 5.1**: Cache keys extended with transpose flags, β!=0 markers, small-N buckets, etc.


## 4) Practical pseudo-code and guidance

4.1 Matmul dispatcher (pseudo-code)

```rust
// Inputs: A [M x K], B [K x N], C [M x N], alpha, beta, dtypes, strides
// Device caps: has_simdgroup_mm
// Goal: choose best kernel per shape and backend availability.

fn matmul_dispatch(params: MatMulParams, caps: DeviceCaps) -> KernelVariant {
    let (m, k, n) = (params.m, params.k, params.n);
    let small_n = n <= 8; // tune thresholds via benchmarks

    if small_n {
        // Prefer specialized GEMV-like kernels
        return KernelVariant::SmallNGemv {
            // choose nx, ny, simdgroup count based on K divisibility and device
            tile_k: pick_tile_k(k, &caps),
            threads_per_row: pick_tpr(n, &caps),
            use_simdgroup: caps.has_simdgroup_mm,
        };
    }

    // Larger GEMM path
    if caps.has_simdgroup_mm && m >= 64 && n >= 16 {
        return KernelVariant::GemmSimdgroup { tile_m: 64, tile_n: 32, tile_k: pick_tile_k(k, &caps) };
    }

    // Fallback to MLX GEMM (preferred over MPS when faster for the shape)
    KernelVariant::MlxGemm { logical_transpose_ok: true, fused_alpha_beta: true }
}
```

4.2 Small-N GEMV kernel sketch (Metal pseudo-code)

```c
// Each simdgroup handles one row or a segment of a row of C
kernel void gemv_small_n(
    device const half* A,  // [M x K]
    device const half* B,  // [K x N], N small
    device half*       C,  // [M x N]
    constant Params&   p,
    threadgroup half*  smemA, // optional tiling of A segments
    threadgroup half*  smemB  // optional tiling of B segments (N small)
) {
    // Strategy:
    // 1. Stage B tiles (K x N) by chunks in threadgroup memory or vector-load directly if cache-friendly.
    // 2. For each row of A assigned to the simdgroup:
    //    - Iterate over K in tiles; accumulate N lanes in registers.
    //    - Use simdgroup operations to share partial sums if needed.
    // 3. Write C row with alpha/beta fusion.
}
```

4.3 Softmax kernels: vector vs block variants

```text
Variant A (vec-softmax): one (or few) simdgroup(s) per row when seq_k <= TG_MAX
- Use simdgroup reduce_max and reduce_add for the row.
- Good for small-to-medium sequence lengths.

Variant B (block-softmax): multiple simdgroups per row (segmented)
- Each segment computes local max/sum; final reduction across segments.
- Good for very long sequence lengths.

Heuristic:
- if seq_k <= 1024 -> vec-softmax (TG size tuned to next pow2 <= seq_k)
- else -> block-softmax with segmented reductions
```

4.4 Logical transpose for K (avoid permute)

```rust
// In QK^T, prefer using stride-transposed K rather than materializing K^T
let k_view = TensorView::with_strides(k_ptr, [S_k, D], [D, 1]); // original
let kt_view = TensorView::with_strides(k_ptr, [D, S_k], [1, D]); // logical transpose
// Pass kt_view into matmul without extra copy when backend supports arbitrary strides.
```

4.5 Fused αAB + βC usage

```rust
// Prefer gemm(alpha, A, B, beta, C) rather than: tmp = A*B; C = alpha*tmp + beta*C
matmul_alpha_beta(A, B, C, alpha, beta, /*transA*/false, /*transB*/maybe)
// Ensure the chosen backend path truly fuses this (MLX path already supports it).
```


## 5) TODO plan for migration

This plan is structured as Quick Wins, Medium Term, and Larger Tasks. Each item should go through the cycle: implement -> test -> benchmark -> integrate (A/B testing) before replacing legacy code. We will maintain toggles to allow safe fallback during rollout.

### 5.1 Quick Wins (1–2 weeks) 
**STATUS: COMPLETED** ✅
- Introduce a shape-aware matmul dispatcher:
  - Add selection logic (as in 4.1) prioritizing: small-N GEMV variant (if available), else MLX GEMM, else MPS.
  - Add environment or feature flags to force specific variants for A/B testing.
- Ensure consistent α/β fuse on all matmul paths:
  - Audit qwen2.5 model code to ensure calls route through fused interfaces.
- Prefer logical transpose for K in SDPA:
  - Route QK^T via logical strides by default when supported (especially through MLX), avoiding K permutes/copies.
- Softmax kernel parameterization:
  - Make threadgroup size a function of seq_k (nearest pow2 up to device limit) with safe bounds and assertions.
  - Introduce a simple simdgroup reduce path when available to reduce barrier count.
- Pipeline cache keys:
  - Extend keys to include: transpose flags, β!=0, small-N bucket, seq_k bucket, causal flag.

### 5.2 Medium Term (2–6 weeks)
- Implement a high-performance small-N GEMV kernel family:
  - N ∈ {1, 2, 4, 8, 16} initial targets; tune tile_k and smem usage.
  - Use simdgroup MM or simdgroup reductions for accumulation where it helps.
  - Benchmark against MLX GEMM and MPS; ensure we surpass MPS for typical inference shapes.
- Softmax variants (vec and block):
  - Implement vec-softmax using simdgroup reductions; handle causal masking efficiently (skip masked tail early).
  - Implement block-softmax with segmented reductions for very long sequence lengths.
  - Validate numerics with strict tolerances and extreme value tests.
- SDPA dispatch heuristics:
  - Choose softmax variant based on (B, head_size, seq_k) buckets.
  - Ensure matmul legs (QK, AV) use the new dispatcher and logical transpose for K.
- Strengthen benchmarking suite:
  - Add matmul shape sweeps for small N and large K; SDPA sweeps for various batch/sequence/head sizes.
  - Integrate metallic_instrumentation labels for per-kernel timing breakdowns.

### 5.3 Larger Tasks (6+ weeks)
- GEMM tiling specialization akin to GGML:
  - Introduce simdgroup MM variants for larger M,N with tuned tile_m/tile_n/tile_k.
  - Add device-capability gating and automatic selection.
- Broader memory-layout optimizations:
  - Eliminate remaining layout transforms and compactions by fully embracing strided views in kernels/backends.
  - Investigate vectorized loads/stores (float2/float4 or half2/half4) and alignment guarantees in APIs.
- Quantization-aware matmul pipelines (future-ready):
  - Design kernel interfaces to plug in Q4/Q5/Q8 pipelines later without changing call sites.
- End-to-end attention pipeline refinement:
  - Consider lightly fused QK-softmax for small sequences (still avoiding full flash), provided it reduces memory traffic without harming clarity.


## 6) Benchmarking and A/B testing plan

- Matmul benchmarks:
  - M×K by K×N with M ∈ {128, 512, 2048}, N ∈ {1,2,4,8,16,32}, K ∈ {1024, 2048, 4096}.
  - Batched variants with realistic leading dimensions and non-compact strides.
  - Compare: current (MLX vs MPS), new small-N GEMV, tuned GEMM.
- SDPA benchmarks:
  - B ∈ {1, 4, 16}, D ∈ {64, 128}, S_q/S_k ∈ {64, 256, 1024, 4096, 8192}.
  - Measure QK, softmax, AV separately; validate correctness against PyTorch with strict tolerances.
- Instrumentation:
  - Use `metallic_instrumentation::gpu_profiler::GpuProfilerLabel` around each kernel.
  - Run on multiple Apple GPUs (M1/M2/M3 families) if available; parameterize heuristics per device family.
- A/B mechanism:
  - Feature flags or env vars to force variant selection.
  - Regression guardrails: disallow merging new variant as default unless it beats current best by threshold (e.g., ≥5%) across primary shapes.


## 7) Risks, guardrails, and design rules

- Defensive programming:
  - Assert threadgroup memory bounds; validate tile divisibility; clear error paths.
- Numerics:
  - Keep accumulation in float for softmax and matmul partials where practical; avoid overflow/underflow.
- Portability:
  - Gate simdgroup usage behind runtime caps; provide functional fallbacks.
- API cleanliness:
  - Keep dispatcher internal but testable; avoid leaking implementation details in public API.
- Performance-first:
  - Minimize copies/permutes; prefer zero-copy strides; avoid unnecessary synchronization.


## 8) Implementation breadcrumbs (paths and ownership)

- Matmul dispatcher and small-N GEMV:
  - Code: dispatcher in `crates/metallic/src/kernels/matmul_dispatcher/` with backends in `matmul_mps/`, `matmul_mlx/`, and `matmul_gemv/`; public invocable `matmul_dispatcher/dispatch_op.rs` created to expose the dispatcher. Tunable thresholds via env: `METALLIC_MATMUL_SMALLN_MAX_N`, `METALLIC_MATMUL_SIMD_M_MIN`, `METALLIC_MATMUL_SIMD_N_MIN`. Backend selection via metallic_env: `FORCE_MATMUL_BACKEND`.
  - Tests/benches: `crates/metallic/src/tests/matmul.rs`, `benches/matmul_dispatcher_bench.rs` (new), `benches/softmax_dispatcher_bench.rs` (new), `benches/gguf_quant_benchmark.rs` (if reusing infra)
- SDPA + softmax variants:
  - Code: `crates/metallic/src/kernels/scaled_dot_product_attention/`, `crates/metallic/src/kernels/softmax/`
  - Tests/benches: `crates/metallic/src/tests/forward_pass_correctness_test.rs`, `benches/sdpa_benchmark.rs`, `benches/sdpa_variant_benchmark.rs`
- Caching and kernel manager:
  - Code: `crates/metallic/src/kernels/kernel_manager.rs`, `crates/metallic/src/resource_cache.rs`
  - Ensure specialization keys include the new factors discussed above.


## 9) Checklists for PRs

Before merging a variant change to default, ensure:
- [ ] Unit and integration tests cover numerics and edge shapes.
- [ ] Benchmarks show ≥5% improvement over current default in primary shapes.
- [ ] No regressions in memory usage or correctness.
- [ ] Feature flags permit easy rollback.
- [ ] cargo fmt, clippy, and cargo build all pass.


---

This plan assumes iterative rollout: implement -> test -> benchmark -> integrate, maintaining legacy paths until data shows advantage. The primary near-term wins are the dispatcher, consistent α/β fusing, softmax tuning with simdgroup reductions, and eliminating permutes via logical strides.


## Appendix A) Low-level matmul kernel design: GGML vs our MLX/MPS/GEMV

This appendix captures concrete, low-level kernel patterns observed in GGML’s Metal matmul paths and contrasts them with our current MLX/MPS usage and our lack of a competitive GEMV. These are actionable in designing custom kernels that can outperform MPS/MLX for transformer inference shapes.

A.1 Data movement and vectorization
- GGML aggressively vectorizes loads/stores:
  - For half/fp16 data, prefer `half2`/`half4` vectorized global loads, mapping to 64/128-bit transactions.
  - Ensure base pointers and row strides are aligned (multiples of 8/16 bytes). If not guaranteed by API, insert prologue that handles misaligned head then switch to vectorized loop.
- For small-N GEMV, stage B’s K×N tile once per threadgroup (N tiny) in threadgroup memory; all rows of A in the group reuse it.
- Use `simdgroup` shuffles/reductions to share partial sums where beneficial instead of extra shared memory and barriers.

A.2 Tiling and threadgroup shape
- GGML chooses tile shapes to match simdgroup/wavefront granularity (e.g., 32-wide) and device limits:
  - Example: tile_m = 64, tile_n = 32, tile_k tuned to keep smem within limits and registers under control.
  - For GEMV small-N: threads-per-row equals next power-of-two >= N, remainder masked; multiple rows per TG for occupancy.
- Dynamic threads-per-threadgroup (nth) chosen as powers-of-two up to kernel max and row length; avoids both underutilization and register oversubscription.

A.3 Accumulation and precision
- Accumulate into float32 even when inputs are half, then convert at epilogue; improves stability and enables better ILP.
- Loop unrolling on K by 2–4 to reduce branch overhead and improve dual-issue; balance with register pressure.

A.4 Barriers and shared memory
- Prefer simdgroup reductions for row-wise max/sum and partial dot products; reduce barrier count dramatically.
- When using shared memory, prefer SoA layouts to avoid bank conflicts and align to 16 bytes; write B tile contiguously by vector lanes.

A.5 Addressing and strides (transpose avoidance)
- GGML passes logical transposes via strides, avoiding permutes. We should consistently:
  - Provide kernels that read non-unit strides efficiently with vectorized loads when stride multiples allow.
  - Fall back to scalar reads only for edge tails.

A.6 Epilogue fusion and alpha/beta
- Epilogue fuses: `D = alpha * Acc + beta * C [+ bias]` with a single global write.
- For small-N, bias/activation (e.g., gelu) can be optionally fused at negligible extra cost compared to a separate pass.

A.7 Occupancy and register pressure
- GGML varies TG size to stay within register limits (observed selection logic for nth and smem). For our kernels:
  - Instrument registers-per-thread via Metal compiler reports and auto-tune a small set of variants.
  - Provide two to three occupancy tiers (e.g., conservative, balanced, aggressive) and select by shape.

A.8 Split-K and batched accumulation (when K large)
- For very large K, consider split-K with partial sums reduced via simdgroup/shared memory; only if it beats the baseline.
- Batched matvec: accumulate multiple rows per TG when M is small to keep the device busy.

A.9 Concrete small-N GEMV kernel sketch (Metal)

```metal
kernel void gemv_n8_f16(
    device const half* A [[buffer(0)]], // M x K, row-major
    device const half* B [[buffer(1)]], // K x N, N<=8
    device half*       C [[buffer(2)]], // M x N
    constant Params&   p [[buffer(3)]],
    threadgroup half*  smemB [[threadgroup(0)]]
) {
    // Threadgroup maps:
    // - tgid.x -> row block of A (e.g., 8 rows)
    // - threads-per-row = 8 or 16 depending on N, next-pow2(N)
    // Stage B: for k in tiles of TK: cooperative load B[k..k+TK, 0..N) into smemB using vectorized half4
    // For each assigned row r: load A[r, k..k+TK) as half4, FMA into float accum[8];
    // After K loop, apply alpha/beta and store C[r, 0..N) with vectorized stores, mask tails
}
```

A.10 MLX vs MPS vs custom
- MLX steel GEMM is strong for general GEMM with good stride support and α/β fusing; however, for small-N GEMV it cannot exploit the N-tiny reuse as effectively as a bespoke kernel.
- MPS often wins when our custom kernel lacks proper vectorization/tiling or when MLX is not tuned for a shape; hence we must beat MPS specifically on the small-N regimes and certain batched layouts.

A.11 Diagnostic checklist for our current underperforming GEMV
- Verify vector width: ensure half2/half4 loads and stores where aligned; avoid scalar loops.
- Ensure B is staged in smem for N<=8 and reused across multiple rows per TG.
- Use float accumulators; unroll the K loop; apply prefetch for the next tile while computing the current one (software pipelining).
- Use simdgroup reductions for any inter-thread accumulation; minimize barriers.
- Validate that the kernel launch fills the GPU (enough rows per TG and enough TGs); avoid too-small grids.

A.12 Additional intrinsics and Metal features to consider
- simdgroup_matrix_multiply for fp16 where supported (M3 and newer), especially for GEMM path.
- quadgroup shuffles (on newer GPUs) to reduce shared mem contention.
- Explicit `threadgroup_barrier(mem_flags::mem_threadgroup)` only when smem is written; use `mem_none` otherwise.


## Appendix B) Additional TODO items (micro-optimizations)

- Implement vectorized IO across kernels:
  - Audit all kernels for half2/half4 usage and add alignment checks/fallbacks.
- Shared memory tiling templates:
  - Provide templates for (TK={64,128}, N={1,2,4,8}) with bank-conflict-free layouts and auto-selected at runtime.
- Loop unrolling and software pipelining for GEMV/GEMM-K loop.
- Auto-tune launch parameters:
  - For each device family (M1/M2/M3), benchmark a small set of TG sizes and tile_k to pick defaults.
- Register pressure instrumentation:
  - Add build flag to dump Metal compiler reports and capture registers-per-thread and spills into CI artifacts.
- Epilogue fusion options:
  - Bias add and optional activation for linear layers where we can remove a pass (with correctness tests).
- Split-K experimental path for very large K.
- Stride-robust vectorization:
  - Implement vector path for common stride patterns (e.g., 2x/4x contiguous) to avoid scalar fallback.
