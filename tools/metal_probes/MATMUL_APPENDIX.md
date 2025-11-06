MATMUL Appendix

This appendix lists kernel variation function names by family with numeric placeholders (###) where parameters vary. Each entry notes its derivation source and the design purpose it was created to test.

**m1_optimized_v7**
- Memory layout note for all `nt_*` kernels: the harness supplies B in `[N x K]` row‑major when `transposeB=true` (NT path). Kernels should index B as `col*K + k` (or via a per‑column pointer) — not `k*N + col`.
- `m1_dot_product_v7_nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128`: Work stealing with CHUNK=4 claims per atomic. Tests contention reduction and column locality; large‑N path alternative to per‑column worksteal.
- `m1_dot_product_v7_nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128`: BN=64 variant of chunked worksteal to increase grid parallelism on wide N shapes.
- `m1_dot_product_v7_nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256`: Large‑K/Small‑N k‑parallel with vec4 loads for both A and B. Tests memory throughput when both operands are vectorized along K.
- `m1_dot_product_v7_nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256`: Same as above with BN=16; two columns per simdgroup. Tests SG utilization vs. per‑column parallelism.
- `m1_dot_product_v7_nt_bn###_largek_smalln_hybrid_kpar_bk###_tg###` (based on `m1_dot_product_v6_nt_bn###_largek_smalln_kpar_bk###_tg###`): Hybrid memory access for large‑K small‑N; simdgroup k‑parallel reduction combined with alignment‑tolerant half4 A loads. Tests reducer overlap, vectorization head/tail, and scalar B strides.
- `m1_dot_product_v7_nt_bn###_col_vec4_worksteal_bk###_tg###` (based on `m1_dot_product_v2_nt_bn###_col_vec4_bk###_tg###`): TG‑local dynamic work distribution; threads claim columns from a shared atomic queue. Tests load balancing for uneven N tiles and L1 reuse without threadgroup memory.
- `m1_dot_product_v7_nt_bn###_smalln_simdgroupmm_tg###` (based on `m1_dot_product_v4_nt_bn###_largek_smalln_tg###`): Small‑N specialization using simdgroup reductions (one column per simdgroup). Tests reducer cost versus dispatch overhead at N ≤ 16.
- `m1_dot_product_v7_nt_bn###_col_vec4_triplebuf_bk###_tg###` (based on `m1_dot_product_v3_nt_bn###_col_vec4_tgread_vA_bk###_tg###`): Triple‑buffered A tiling to overlap loads and compute via a ring. Tests pipeline latency hiding and barrier placement sensitivity.
- `m1_dot_product_v7_nt_bn###_fused_bias_relu_tg###` (based on `m1_dot_product_v7_sbcast_vec4_bn###_tg###`): Fused matmul + bias + ReLU epilogue. Tests writeback fusion cost and branchless activation throughput in fp16.
- `m1_dot_product_v7_nt_bn###_fused_bias_gelu_tg###` (based on `m1_dot_product_v7_sbcast_vec4_bn###_tg###`): Fused matmul + bias + GELU epilogue. Tests transcendental activation performance and numerical stability under fp16.
- `m1_dot_product_v7_nt_adaptive_bn###_tg###` (based on `m1_dot_product_v2_nt_bn###_col_bk###_tg###` and v6 k‑parallel): In‑kernel adaptive path selection; chooses k‑parallel reduction for large‑K/small‑N else column‑tiling. Tests heuristic thresholds and mixed reducer/tiled cost.
- `m1_dot_product_v7_nt_bn###_col_vec4_transformB_bk###_tg###` (based on `m1_dot_product_v2_nt_bn###_col_vec4_bk###_tg###`): Transforms B submatrix into threadgroup memory for contiguous access. Tests coalescing improvements versus shared‑memory footprint and barrier overhead.
- `m1_dot_product_v7_sbcast_vec4_bn###_tg###` (standalone baseline): A‑tiling‑only column path; tests occupancy and threadgroup memory footprint without B tiling.
- `m1_dot_product_v7_kpar2_bn###_tg###` (standalone baseline): Two‑stage k‑parallel reduction; tests reducer correctness and harness integration for large‑K/small‑N.

**m1_optimized_v6**
- m1_dot_product_v6_nt_bn###_col_vec4_tgread_vA_bk###_tg### (based on m1_dot_product_v3_nt_bn###_col_vec4_tgread_vA_bk###_tg###): Large-N path; vec4 B loads with A staged in threadgroup memory, testing staging overhead, double-buffering, and memory coalescing.
- m1_dot_product_v6_nt_bn###_largek_smalln_kpar_bk###_tg### (based on m1_dot_product_v4_nt_bn###_largek_smalln_tg###): Large-K/Small-N path; k-parallel reduction with hierarchical single-barrier final reduce, testing BK/TG occupancy trade-offs and reducer contention.

**m1_optimized_v5**
- m1_dot_product_v5_nt_ultra_tiny_bn###_tg### (based on m1_dot_product_v4_nt_tiny_bn###_tg###): Ultra-low overhead path for very small shapes; no threadgroup memory, manual unrolling, testing whether overhead removal wins at tiny sizes.
- m1_dot_product_v5_nt_bn###_largek_smalln_tg### (based on m1_dot_product_v4_nt_bn###_largek_smalln_tg###): Aggressive large-K/Small-N specialization; tests reducer tuning and threadgroup sizing for occupancy improvements.
- m1_dot_product_v5_nt_dbg_largek_smalln_single_tg### (based on m1_dot_product_v4_nt_bn###_largek_smalln_tg###): Debug single-column accumulation variant; validates reducer correctness and scheduling under minimal concurrency.

**m1_optimized_v4**
- m1_dot_product_v4_nt_bn###_col_vec4_tgread_bk###_tg### (based on m1_dot_product_v3_nt_bn###_col_vec4_tgread_bk###_tg###): Large-N vectorized path; adds threadgroup staging for A tile, testing BK depth and TG sizing effects on throughput.
- m1_dot_product_v4_nt_tiny_bn###_tg### (based on m1_dot_product_v2_nt_bn###_col_bk###_tg###): Lightweight NT path for tiny shapes; tests benefit of reduced overhead versus shared-memory use.
- m1_dot_product_v4_nt_bn###_largek_smalln_tg### (based on m1_dot_product_v2_nt_bn###_row): First k-parallel reduction attempt for K≫N; tests reducer structure and row-major access impact.

**m1_optimized_v3**
- m1_dot_product_v3_nt_bn###_col_vec4_sgbr_bk###_tg### (based on m1_dot_product_v2_nt_bn###_col_bk###_tg###): NT path with simdgroup broadcast of the A tile and half4 B loads; tests warp-level reuse and register pressure trade-offs.
- m1_dot_product_v3_nt_bn###_col_vec4_tgread_bk###_tg### (based on m1_dot_product_v2_nt_bn###_col_vec4_bk###_tg###): NT path that reads A from threadgroup memory; tests shared-memory footprint versus load coalescing.
- m1_dot_product_v3_nt_bn###_col_vec4_tgread_vA_bk###_tg### (based on m1_dot_product_v3_nt_bn###_col_vec4_tgread_bk###_tg###): NT path staging A explicitly in threadgroup memory; tests pointer arithmetic overhead and staging benefits.
- m1_dot_product_v3_nt_bn###_col_vec4_tgread_unroll16_bk###_tg### (based on m1_dot_product_v3_nt_bn###_col_vec4_tgread_bk###_tg###): NT path with inner-loop unroll factor 16; tests ILP and register usage effect on throughput.
- m1_dot_product_v3_nt_bn###_col_vec4_tgread_vA_unroll16_bk###_tg### (based on m1_dot_product_v3_nt_bn###_col_vec4_tgread_vA_bk###_tg###): NT path combining A staging and unroll 16; tests pipeline overlap and stall behavior.

**m1_optimized_v2**
- m1_dot_product_v2_nt_bn###_col_bk###_tg### (based on m1_dot_product_tiled_simd##): Baseline NT column-major kernel; tests simple tiling and occupancy with varying BK/TG.
- m1_dot_product_v2_nt_bn###_col_vec4_bk###_tg### (based on m1_dot_product_v2_nt_bn###_col_bk###_tg###): Vectorized B loads (half4); tests memory coalescing, alignment, and lane utilization.
- m1_dot_product_v2_nt_bn###_col_bt_bk###_tg### (based on m1_dot_product_v2_nt_bn###_col_bk###_tg###): B-tiling to threadgroup memory; tests shared-memory reuse versus synchronization overhead.
- m1_dot_product_v2_nt_bn###_row (based on m1_dot_product_v2_nt_bn###_col_bk###_tg###): Row-major indexing; tests stride patterns and cache locality effects.
- m1_dot_product_v2_tiled## (standalone): Tiled dot-product prototypes (## ∈ {1,2,4}); tests increasing tile factors with simdgroup utilization.

**m1_dot_product (baseline prototypes)**
- m1_dot_product_tiled_simd## (standalone): Baseline tiled dot-product kernels (## ∈ {1,2,4}); tests simdgroup matrix ops vs software fallback tiling.

**original_mlx**
- gemm_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (MLX baseline): General GEMM; tests MLX’s tiling and data layouts across transposes and tile shapes.
- gemm_bias_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (MLX baseline): GEMM with bias; tests epilogue fusion impact and memory access with bias.

**optimized_mlx_bk64**
- gemm_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): GEMM tuned toward deeper BK=64; tests K-tiling depth effects on occupancy and cache behavior.
- gemm_bias_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): Bias epilogue variant with BK=64 emphasis; tests fused epilogue cost under deeper K tiles.

**optimized_mlx_m1**
- gemm_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): M=1–focused tuning; tests single-row throughput and tile aspect ratio adjustments.
- gemm_bias_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): Bias epilogue for M=1; tests fusion and writeback under thin-M regimes.

**optimized_mlx_nn**
- gemm_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): NN-layout specialization; tests transpose handling and pointer math for NN shapes.
- gemm_bias_<T>_<INAME>_<ONAME>_<BM>_<BN>_<BK>_<WM>_<WN> (based on original_mlx): NN-layout with fused bias; tests epilogue cost and memory locality.

**original_gemm_tiled**
- gemm_tiled_f16 (standalone): Experimental tiled GEMM; tests hardware SIMD-group MMA path vs software threadgroup tiling fallback and epilogue blending.

**original_gemv**
- gemv_f32 (standalone): Generic GEMV (f32); tests shared-memory staging of B tiles and dot-product accumulation in single-precision.
- gemv_f16 (standalone): Generic GEMV (f16); tests shared-memory staging and half-precision accumulation behavior.

**original_gemv_smalln**
- gemv_n1_f16 (standalone): N=1 specialization; tests single-column throughput with cooperative B tiling.
- gemv_n2_f16 (standalone): N=2 specialization; tests minimal-width vectors and float2 accumulation.
- gemv_n4_f16 (standalone): N=4 specialization; tests small-width vectorization and shared-memory reuse.
- gemv_n8_f16 (standalone): N=8 specialization; tests balanced TG sizes and cooperative loads for moderate N.
- gemv_n16_f16 (standalone): N=16 specialization; tests wider output columns with cooperative shared-memory loads.
