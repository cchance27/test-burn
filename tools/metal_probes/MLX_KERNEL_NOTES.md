## MLX Matmul Probe Status

### High-Value Shapes

- **m=1, n=9728, k=896 (NT)**  
  - `mlx/m1_fast` (BM=8, BN=128, BK=64) now matches MPS on GPU (0.161 ms) but CPU round trip is ~0.50 ms (+54% vs 0.325 ms production).  
  - `mlx/bk64` and `nn_transposed` regress.  
  - GEMM tiled baseline beats generic MLX on CPU (0.20 ms vs 0.44 ms), but not on GPU.  
  - Best GPU: tie between `mps/baseline` and `mlx/m1_fast` at ~0.161 ms.

- **m=1, n=896, k=4864 (NT)**  
  - MPS remains far ahead (0.086 ms GPU).  
  - `mlx/bk64` is the best MLX variant (~0.384 ms GPU, 0.657 ms CPU).  
  - `m1_fast` offers no improvement; still ~0.48 ms GPU.  
  - Large-GEMV behavior dominates here.

- **m=1, n=896, k=896 (NT)**  
  - MPS ~0.025 ms GPU.  
  - MLX variants range 0.08–0.11 ms GPU.  
  - `m1_fast` roughly matches baseline here.

- **m=1, n=1152, k=896 (NN, bias)**  
  - MPS dominates at 0.033 ms GPU.  
  - MLX variants: 0.084–0.117 ms GPU, CPU 0.26–0.49 ms.  
  - `nn_transposed` didn’t help.

- **m=1, n=151936, k=896 (NT)**  
  - `gemm_tiled` baseline is best (0.586 ms GPU).  
  - MPS and MLX variants ~2.0 ms GPU.

### Takeaways

1. **MLX M=1 Fast Path**: GPU parity with MPS achieved, but CPU time remains high (0.50 ms). Need command-buffer optimizations before production.  
2. **Large-K Skinny-N**: MLX (and GEMM tiled) still 4–6× slower than MPS; investigate dot-product path or weight layout changes.  
3. **NN Transpose Experiment**: Feeding pre-transposed weights (`nn_transposed`) didn’t help; overhead lies elsewhere.  
4. **Bias Add Cases**: MPS remains best; MLX requires a dedicated epilogue optimization.
5. **GEMV Coverage Gap**: Our GEMV kernels only support bias-free workloads, so the enhanced harness routes bias shapes elsewhere. Addressing this requires kernel-side epilogue work before we can flip the JSON flags.
6. **Enhanced Harness Flow**: The new Swift harness loads `variants_enhanced.json`, applies per-variant support filtering, and skips unsupported specs up front while still recording the gap. This keeps the run list realistic and documents missing coverage.

### Current MLX Variants (`matmul/variants.json`)

- `mlx/baseline` → `original_mlx`
- `mlx/m1_fast` → `optimized_mlx_m1` (BM=8, BN=128, BK=64)
- `mlx/bk64` → `optimized_mlx_bk64` (BM=32, BN=32, BK=64)
- `mlx/nn_fast` → `optimized_mlx_nn` (BM=32, BN=64, BK=32)
- `mlx/nn_transposed` → `optimized_mlx_nn` + `transposeBOverride=false`
- `mps/baseline`
- `gemv/baseline`, `gemv/smalln`
- `gemm_tiled/baseline`

All MLX variants now build with override tiles and optional transpose handling via the Swift harness (`run_matmul_probes.swift` / `run_matmul_probes_enhanced.sh`). `variants_enhanced.json` captures capability flags (`supports.*`) such as `smallK`, `bias`, `accumulate`, and `supportedNValues`. These drive the new filtering logic, so updating a kernel’s feature set now means (1) fixing the Metal code, (2) widening the guard predicates, and (3) flipping the JSON booleans.

### Next Experiments

**NOTE**: We've uploaded a toools/metal_probes/dot_product_example_from_mlx.metal for reference on a supposedly very fast mlx implementation of dot product on metal kernels from the mlx project, so maybe its helpful to us.


1. **Custom m=1 Dot-Product Kernel**  
   - Create a bespoke kernel without threadgroup staging/epilogue for pure matmul (BN=128, BK=64).  
   - Goal: keep GPU 0.16 ms but cut CPU towards 0.30 ms.

2. **Large-K GEMV Hybrid**  
   - Design a dot-product style tiler for `m=1, k>>n`.  
   - Borrow ideas from MPS’ pay-as-you-go dot product or TMA (Tensor Memory Accelerator) style loads.

3. **Command-Buffer Profiling**  
   - Use GPU counters / Instruments to isolate where the extra 0.15–0.20 ms CPU time is spent for `m1_fast`.  
   - Experiment with batching multiple dispatches per command buffer in the harness.

4. **Bias/Epilogue Fusion**  
   - Craft BK=64 variant with fused bias/AXPBY epilogue to test bias-add shapes.

### Reference Commands

```bash
MODULE_CACHE_DIR="$PWD/.module_cache" MATMUL_BENCH_ITERS=10 METAL_RUN=1 ./run_matmul_probes.sh
```

Files of interest:
- `tools/metal_probes/matmul/optimized_mlx_m1.metal`
- `tools/metal_probes/matmul/optimized_mlx_bk64.metal`
- `tools/metal_probes/matmul/optimized_mlx_nn.metal`
- `tools/metal_probes/run_matmul_probes.swift`
- `tools/metal_probes/matmul/variants.json`

---

## 2025-11-03 — M=1 NT v2 Kernel Status (m1_optimized_v2)

Summary of the latest clean run and kernel work focused on m=1, transposeB=true (NT) shapes.

What worked
- A-tiling + double buffer + 2 columns/thread: Correct and fast across shapes.
- Vectorized B loads (half4) with deeper unroll: Clear gains on hot shapes.
- bn64 + tg128 is the best general mapping for Qwen-like m=1 shapes; bn128 is close but usually second.
- New v3 SIMD-group-broadcast path mirrors the bn64/bn128 vec4 geometry but replaces per-thread TG reads with `simd_broadcast_first` so only one lane touches threadgroup memory per SIMD. Code landed as `m1_dot_product_v3` and is ready for benchmarking.
- Correctness stable across all variants (maxRel ~ 4.8e-4), including tails for K not multiple of unroll.

Best results observed
- m=1, n=9728, k=896 (NT): `nt_bn64_col_vec4_bk128_tg128` at 0.142 ms GPU — faster than MPS (0.162 ms).
- m=1, n=151936, k=896 (NT): `nt_bn64_col_vec4_bk64_tg128` at 2.038 ms GPU — ~4% from MPS (1.961 ms).
- m=1, n=896, k=896 (NT): MPS still best at 0.026 ms; our best (`nt_bn128_col_vec4_bk128_tg128`) ~0.075–0.08 ms. MLX bk64_nt remains very competitive here (~0.036 ms).

What didn’t help
- B-tiling (staging BK×BN slice of B in threadgroup memory) regressed performance on all tested shapes. We disabled all `*bt*` variants by default in `variants_enhanced.json` to keep runs focused.
- Earlier tg64 versions missed half the columns due to barrier placement; fixed by (1) mapping 2 columns per thread and (2) making all barriers uniform across the TG.

Design/implementation notes
- Dispatch uses variant name tokens (`bnXX`, `tgYY`) to configure per-variant geometry — this is not a runtime heuristic; the harness stays neutral. Heuristics will live in Metallic, not the harness.
- Vectorized path handles alignment: scalar head steps to 4-aligned K, runs half4 core (unroll=8), then scalar tail.
- Register use remained within budget; threadgroup memory stayed under typical limits (e.g., BK=128 uses ~32 KB for B when we tested, but B-tiling is now disabled).

Recommendations
- Default benchmark set (enabled): A-tiling `*_col_*` + vectorized `*_col_vec4_*`. Keep bn64+tg128 as primary variants to compare against MPS and MLX.
- Disabled by default: all `*_col_bt_*` (B-tiling) variants.
- Added a new backend (`m1_optimized_v3`) that contains the SIMD-broadcast vec4 variants; run these side-by-side with v2 to isolate the benefit of removing redundant TG memory reads.

Ideas to test next
- Increase columns-per-TG: Try `bn256` with tg128 (2 cols/thread) for very large N to reduce TG count.
- Wider vector loads: half8 (two half4) where alignment guaranteed; keep scalar head/tail guards.
- SIMD-group broadcast benchmarking: the code exists in v3; next step is to capture perf deltas vs. v2 and confirm TG memory bandwidth relief on large-K cases.
- ILP tuning: explore unroll=16 for vec path, balancing register pressure vs occupancy.
- Occupancy sweeps: revisit tg64 vs tg128 across BN=64/128 with the vec4 path.

Harness policy reminder
- The benchmark harness remains heuristic-free by design; we only map dispatch geometry from variant names and filter by declared supports. Any selection heuristics will be implemented in the Metallic framework, not in the harness.

---

## 2025-11-04 — M=1 NT v3 Progress (sgbr / tgread / tgread_vA)

Source status
- v3 kernel families in `m1_dot_product_v3.metal`:
  - `sgbr`: simdgroup-broadcast of A tile (only one lane touches TG per SIMD); vectorized B (half4).
  - `tgread`: per-lane TG reads for A (Apple HW often broadcasts); vectorized B.
  - `tgread_vA`: manual vectorization of A from TG (scalar pack → float4) + vectorized B.
  - `unroll16`: deeper ILP variant of `tgread`/`tgread_vA` for tg128.
  - tg64 occupancy variants exist for tgread and vA.

Dispatch fix
- Fixed Swift dispatch parsing to extract `bnXX`/`tgYY` only when followed by digits. This avoids falsely matching `tg` inside the word `tgread` and launching 128-thread groups for `tg64` kernels. After this change, all tg64 variants launched correctly and no longer returned 0.0ms timings.

Correctness
- All v3 variants now validate with maxRel ≈ 4.8e-4 on the hot shapes. Earlier large-error artifacts (and 0.0ms) were caused by the misdispatch bug, not kernel math.

Performance observations (GPU time)
- m=1, n≈10k, k=896 (NT):
  - Best v3: `nt_bn64_col_vec4_tgread_bk128_tg128` ≈ 0.142 ms (close to MPS 0.134 ms).
  - `sgbr` is similar (≈0.172–0.175 ms). `tg64` is slower (≈0.165–0.191 ms).
  - `vA` does not improve over `tgread` and often adds CPU overhead.
- m=1, n=896, k=4864 (NT):
  - Best v3: `nt_bn128_col_vec4_tgread_bk64_tg128` ≈ 0.196 ms (still >2× MPS 0.085 ms).
  - `tg64` regresses badly (e.g., ≈0.45–0.78 ms); `vA` also regresses.
- m=1, n=896, k=896 (NT):
  - Best v3 tg128 ≈ 0.072–0.082 ms; MPS still best at ≈0.024 ms.
  - `tg64` slower than tg128 here as well (≈0.141 ms vs 0.074 ms).

Takeaways
- tg128 is the right occupancy for the v3 designs across the hot shapes. tg64 should be kept for targeted experiments only.
- `tgread` is a solid baseline; `sgbr` is close and sometimes slightly worse/neutral. `tgread_vA` adds packing cost without consistent wins.
- `unroll16` offers tiny gains on n≈10k but regresses on large-K skinny-N; keep only where it wins.
- CPU timings remain higher than MPS; addressable via harness-side batching/reuse (outside kernel scope).

Action items
- Keep enabled in default sweeps: tg128 (`tgread`, `sgbr`), with BN in {64, 128}, BK in {64, 128}. Keep one `tgread_vA` as control.
- Disable by default (retain for opt-in tests): tg64 across the board; `unroll16` for large-K cases.
- Next kernel probes: refine ILP for large-K via software pipelining of B loads, confirm barrier placement minimality, and evaluate half8 (two half4) once we strengthen alignment guards.

Disabled variants (default sweeps)
- sgbr: disabled `nt_bn128_col_vec4_sgbr_bk128_tg128`, `nt_bn64_col_vec4_sgbr_bk128_tg128`, `nt_bn64_col_vec4_sgbr_bk64_tg128`; kept `nt_bn128_col_vec4_sgbr_bk64_tg128` only.
- tg64 occupancy: disabled `nt_bn128_col_vec4_tgread_bk64_tg64`, `nt_bn64_col_vec4_tgread_bk64_tg64`.
- vA paths: disabled all tg128 (`nt_bn128/64_col_vec4_tgread_vA_*_tg128`) and all tg64 (`*_tgread_vA_*_tg64`).
- unroll16: disabled tgread and tgread_vA (`*_tgread_unroll16_bk64_tg128`, `*_tgread_vA_unroll16_bk64_tg128`).
- bn256 (sgbr): disabled both `nt_bn256_col_vec4_sgbr_bk{64,128}_tg128` by default pending validation.
