# MLX Kernel Optimization Comprehensive Plan

## Overview

This plan leverages insights from the MLX dot product example to optimize Metal kernels for the Qwen2.5 shapes, addressing the key bottlenecks identified in the current implementation while maintaining GPU performance parity with MPS.

### Current Harness & Manifest State

- The enhanced Swift harness (`run_matmul_probes_enhanced.sh` pipeline) now loads `variants_enhanced.json`, enforces each variant’s `supports.*` flags, and records any filtered specs as explicit gaps.  
- Updating kernel capabilities therefore requires flipping the matching JSON booleans (`smallK`, `bias`, `accumulate`, `supportedNValues`, etc.) after the Metal implementation is proven.  
- GEMV variants deliberately advertise `bias: false` today; until we land epilogue/bias support in Metal we should leave those specs routed to other backends.
- All `_col_bt_` (B-tiling) variants stay present but `enabled:false` in the manifest after repeated regressions; targeted experiments can flip them on, but default sweeps stay focused on the faster A-tiling paths.  
- Introduced a dedicated backend (`m1_optimized_v3`) that houses the SIMD-group-broadcast vec4 variants. The harness derives dispatch geometry from variant names (`bnXX`, `bkYY`, `tgZZ`) exactly as it does for v2, keeping heuristics out of the probes.

## Key Insights from MLX Dot Product Example

The MLX dot product example provides several optimization patterns that can be adapted:

1. **SIMD Group 8x8 Operations**: Efficient simdgroup matrix operations with proper loading and storing
2. **Tiled Memory Access**: Block tiling strategies with shared memory caching
3. **Memory Coalescing**: Techniques to ensure coalesced global memory access
4. **Register Optimization**: Strategic use of register caching for intermediate results
5. **Threadgroup Synchronization**: Proper use of barriers for data consistency

## Current Performance Status

- **Achievement**: `m1_fast` kernel achieves GPU parity with MPS (~0.161ms) for m=1, n=9728, k=896
- **Bottleneck**: High CPU overhead (~0.50ms) due to command buffer management
- **Critical Gap**: Large-K shapes (like m=1, n=896, k=4864) are 4-6x slower than MPS
- **Areas for Improvement**: Bias-add operations, GEMV performance, and memory access patterns

## Phase 1: High-Impact Optimizations (Immediate Focus)

### 1. Custom M=1 Dot-Product Kernel with MLX Patterns
- **Goal**: Maintain GPU performance (~0.16ms) while reducing CPU overhead from ~0.50ms to ~0.30ms
- **Strategy**: 
  - Implement optimized SIMD group 8x8 operations from MLX example
  - Use efficient loading patterns to minimize memory latency
  - Remove unnecessary threadgroup staging for pure matmul cases
- **Based on**: `sgemm_naive_simd` and `sgemm_tiled_simd` patterns from MLX
- **Target**: m=1, n=9728, k=896 shape optimization

### 2. Large-K GEMV Hybrid with Tiled Approach
- **Goal**: Close the 4-6x performance gap for large-K skinny-N shapes
- **Strategy**:
  - Adapt the `sgemm_1d_block_tiling` and `sgemm_2d_block_tiling` patterns
  - Implement dot-product style processing for k>>n cases
  - Use shared memory blocking to maximize cache efficiency
- **Manifest Follow-up**: Once bias handling exists for GEMV, flip `supports.bias` to `true` in `variants_enhanced.json` so the harness stops filtering those specs.
- **Based on**: Block tiling from MLX example with optimized strides
- **Target**: m=1, n=896, k=4864 and similar shapes

### 3. Command-Buffer Optimization
- **Goal**: Reduce CPU overhead through batching and reuse
- **Strategy**:
  - Batch multiple dispatches per command buffer
  - Implement command buffer reuse patterns
  - Profile with Instruments to identify bottlenecks
- **Integration**: Add to Swift harness for benchmarking

## Phase 2: Broad-Spectrum Enhancements

### 4. SIMD Group Optimization Based on MLX Patterns
- **Goal**: Optimize for Apple's SIMD architecture (32 threads per SIMD group)
- **Strategy**:
  - Implement efficient `simdgroup_multiply_accumulate` operations
  - Use proper `simdgroup_load` and `simdgroup_store` patterns
  - Optimize for 8x8 matrix operations (matching MLX reference)
- **Based on**: `sgemm_tiled_simd` and `sgemm_naive_simd` kernels
- **Implementation**: Adapt the NBLK template system for different tile sizes
- **Status**: The first SIMD-broadcast implementation (`m1_dot_product_v3`) is available for bn64/bn128 × bk64/bk128 (tg128). Next step is benchmarking vs. the v2 vec4 kernels and validating gains on large-K shapes.

### 5. Memory Access Optimization
- **Goal**: Implement coalesced memory access patterns and eliminate bank conflicts
- **Strategy**:
  - Use the coalescing patterns from `sgemm_coalescing` in MLX
  - Implement proper stride patterns to maximize bandwidth
  - Optimize shared memory layout to avoid bank conflicts
- **Based on**: Memory access patterns from the MLX example

### 6. Shared Memory Caching Strategy
- **Goal**: Optimize threadgroup memory usage for cache efficiency
- **Strategy**:
  - Adapt the `sgemm_shared_mem_block` pattern for efficient caching
  - Implement blocking factors that match the MLX efficient patterns
  - Optimize A and B matrix caching with proper synchronization
- **Based on**: `sgemm_shared_mem_block` kernel from MLX

## Phase 3: Advanced Optimizations

### 7. Shape-Specific Kernels with MLX Patterns
- **Goal**: Create specialized kernels for different shape categories using MLX techniques
- **Strategy**:
  - Small-N kernels (n <= 16) using vectorized patterns
  - Large-K kernels using tiled approaches from MLX
  - Square/rectangular optimization with SIMD efficiency
- **Based on**: Adaptation of `sgemm_tiled_simd` variants with NBLK templates

### 8. MLX-Inspired TMA Loading Patterns
- **Goal**: Implement advanced prefetching using MLX load/store patterns
- **Strategy**:
  - Use `simdgroup_load`/`simdgroup_store` for overlapping computation
  - Implement progressive loading strategies from MLX examples
  - Optimize memory bandwidth utilization with proper buffering
- **Based on**: SIMD group operations from MLX example

### 9. Dynamic Kernel Selection
- **Goal**: Automatically select optimal kernels per shape using MLX-based variants
- **Strategy**:
  - Implement runtime decision logic based on M/N/K dimensions
  - Add performance prediction models using MLX patterns
  - Integrate with existing Swift harness for benchmarking
- **Implementation**: Extend `MatmulHarness` to select from MLX-inspired kernels

## Phase 4: Specialized Optimizations

### 10. Bias/Epilogue Fusion with MLX Patterns
- **Goal**: Improve bias-add performance using SIMD group operations
- **Strategy**:
  - Implement fused bias addition during SIMD computation
  - Use register caching patterns from MLX example (`regM[TM]`, `regN[TN]`)
  - Minimize extra memory passes with inline operations
- **Target**: m=1, n=1152, k=896 bias-add shapes

### 11. Advanced Tiling with Register Caching
- **Goal**: Optimize register usage for maximum computational efficiency
- **Strategy**:
  - Implement register caching as seen in `sgemm_2d_block_tiling`
  - Use `threadResults[TM * TN]` patterns for intermediate storage
  - Balance register pressure with computational efficiency
- **Based on**: Register optimization patterns from MLX example

## Phase 5: Validation and Integration

### 12. Testing and Validation Framework
- **Goal**: Maintain comprehensive correctness with new MLX-optimized kernels
- **Strategy**:
  - Implement precision validation for all MLX-based variants
  - Add regression benchmarks for all critical shapes
  - Integrate with existing metallic crates seamlessly
  - Maintain error tolerance checks from current implementation

### 13. Performance Monitoring Dashboard
- **Goal**: Track improvements and regressions of MLX-based kernels
- **Strategy**:
  - Create tooling to compare MLX vs original implementations
  - Automated comparison against MPS baselines
  - Monitor both CPU and GPU performance metrics
  - Identify performance regressions automatically

## Implementation Timeline

### Week 1: CPU Overhead Reduction
- Command-buffer optimization
- Initial MLX SIMD patterns for M=1 kernels

### Immediate Next Steps (rolling)
- Benchmark the new v3 SIMD-broadcast variants (bn64/bn128 × bk64/bk128, tg128) against the v2 vec4 kernels and MPS baselines; capture both GPU/CPU deltas.
- Prototype bn256 column tiles (tg128; still 2 columns/thread) to see if fewer threadgroups help huge-N workloads (>150k).
- Explore half8 vector loads (two half4) once alignment guarantees are in place; keep scalar head/tail guards for leftovers.
- Experiment with deeper ILP (unroll=16) on the vec path and monitor register pressure / spills.
- Revisit tg64 vs. tg128 for the vec and broadcast paths to understand occupancy/latency trade-offs per spec.
- DEBT: v3 tg64 + vA (vectorized A from TG) variants are temporarily disabled in `variants_enhanced.json` due to precision mismatches and suspicious 0.0ms GPU timing in the harness. Re‑audit kk‑relative indexing and tail/align guards for packed half4 B loads under tg64 occupancy; re‑enable once correctness matches baseline (maxRel ≤ ~4.8e-4).

---

## 2025-11-04 — v3 Progress Snapshot and Plan Updates

Summary of latest run (iterations=6)
- Correctness: All v3 variants pass with maxRel ≈ 4.8e-4 after dispatch fix (bn/tg token parsing). The prior 0.0ms + huge-error artifacts are resolved.
- m=1, n=9728, k=896: Best v3 ~0.142 ms (tgread, bn64, bk128, tg128). MPS remains best at ~0.134 ms.
- m=1, n=896, k=4864: Best v3 ~0.196 ms (tgread, bn128, bk64, tg128), still >2× MPS (0.085 ms). tg64 variants regress heavily here.
- m=1, n=896, k=896: Best v3 ~0.072–0.082 ms (tg128). MPS is ~0.024 ms.

Decisions (default sweep set)
- Keep: tg128 variants (`tgread`, `sgbr`) with BN∈{64,128} and BK∈{64,128}. For sgbr, keep `bn128/bk64` only.
- Disable: `tgread_vA` (all tg128), all tg64 variants (`tgread` and `vA`), and all `unroll16` (both `tgread` and `vA`) in default sweeps.
- Experimental only: re‑enable any of the above selectively if targeting a specific micro-study.

Root‑cause fix captured
- Swift dispatch now extracts `bnXX`/`tgYY` only when followed by digits, avoiding accidental `tg` match inside `tgread`. This was the source of earlier tg64 mis‑launches.

Next kernel work (prioritized)
- Large‑K skinny‑N focus: Improve ILP and memory pipelining in the `tgread` path with software prefetch and reduced dependency chains; target `bn128/bk64/tg128` as current winner to beat.
- Alignment and vectorization: Evaluate half8 (two half4 reads) where K‑alignment permits; keep solid scalar head/tail guards and ensure no misaligned half4 casts.
- Barrier minimization: Audit all `threadgroup_barrier` placements to ensure only the strictly necessary barriers remain (especially around double‑buffered A prefetch), preserving correctness.
- SGBR tuning: Confirm bank‑conflict behavior and whether simdgroup broadcast helps on very large‑N; keep as a near‑parity alternative.

Harness work (non‑kernel)
- Reduce CPU overhead by batching more dispatches per command buffer and by reusing encoders where safe. This will not affect GPU times but improves host‑side latency comparisons against MPS.

Disabled set (reflected in variants_enhanced.json)
- sgbr: `nt_bn128_col_vec4_sgbr_bk128_tg128`, `nt_bn64_col_vec4_sgbr_bk128_tg128`, `nt_bn64_col_vec4_sgbr_bk64_tg128` (kept `nt_bn128_col_vec4_sgbr_bk64_tg128`).
- tg64: `nt_bn128_col_vec4_tgread_bk64_tg64`, `nt_bn64_col_vec4_tgread_bk64_tg64`.
- vA: all tg128 (`nt_bn128/64_col_vec4_tgread_vA_*_tg128`) and all tg64 (`*_tgread_vA_*_tg64`).
- unroll16: `*_tgread_unroll16_bk64_tg128`, `*_tgread_vA_unroll16_bk64_tg128`.
- bn256 sgbr: `nt_bn256_col_vec4_sgbr_bk{64,128}_tg128` kept disabled pending separate validation.

### Week 2: Large-K GEMV Hybrid
- Implement block tiling from MLX example
- Optimize for critical large-K shapes

### Week 3: SIMD Group Optimization
- Full 8x8 SIMD group implementation
- Memory coalescing improvements

### Week 4: Bias Fusion
- Fused bias-add operations with MLX patterns

### Week 5-6: Advanced Optimizations
- Shape-specific kernels
- Register caching improvements

### Week 7+: Integration and Validation
- Comprehensive testing
- Performance monitoring

## Success Metrics

1. **CPU Overhead**: Reduce from ~0.50ms to <0.30ms for m1_fast kernel
2. **Performance Gap**: Reduce large-K performance gap to <2x of MPS
3. **GPU Performance**: Maintain or improve current GPU performance
4. **Coverage**: Support all critical Qwen2.5 shapes with competitive performance
5. **MLX Baseline**: Show clear improvements over original MLX implementation

## Technical Implementation Notes

### SIMD Group Operations Best Practices (from MLX)
- Use 8x8 tile sizes to match SIMD width
- Proper barrier synchronization between load/compute/store phases
- Efficient load/store patterns for matrix data
- Accumulation patterns that maximize ALU utilization

### Memory Access Optimizations (from MLX)
- Coalesced access patterns by organizing threads appropriately
- Shared memory blocking with optimal tile sizes
- Stride patterns that maximize memory bandwidth
- Bank conflict avoidance in threadgroup memory

### Template System for Flexibility (from MLX)
- NBLK template system for different block sizes
- Configurable BM, BN, BK, TM, TN parameters
- Compile-time optimization for different shape categories

This plan leverages the proven MLX optimization patterns while addressing the specific performance bottlenecks in the current implementation. The step-by-step approach ensures measurable progress while maintaining code quality and correctness.

---

## 2025-11-03 — Current Plan Checkpoint (m=1, NT)

Completed
- Implemented v2 m=1 NT kernels with A-tiling + double buffer and 2 cols/thread mapping.
- Fixed tg64 coverage and barrier placement; correctness validated (maxRel ~ 4.8e-4).
- Added vectorized B-loads (half4) variants with deeper unroll; enabled in variants.
- Disabled all B-tiling variants (`*_col_bt_*`) in `variants_enhanced.json` due to regressions.
- Verified dispatch is variant-driven (bnXX/tgYY tokens), keeping the harness heuristic-free.

Observations
- bn64 + tg128 variants generally outperform bn128 across Qwen-25 shapes.
- BK=128 best for moderate N (~10k), BK=64 best for very large N (~150k).
- B-tiling adds overhead on Apple GPUs for these shapes; A-tiling alone + vec loads is superior.

Near-term tasks
- Add `bn256_*` variants for very large N and compare vs bn64.
- Explore simdgroup broadcasts for staged A to reduce shared-memory traffic. **(Started: implemented in v3; pending perf data.)**
- Try half8 vectorized loads (two half4) where alignment is safe; keep scalar head/tail for edges.
- Tight ILP/occupancy sweep: unroll 8 vs 16 on vec path, confirm register usage and spill-free execution.
- Keep bt_* variants disabled; leave them in the manifest for reference/testing only.

Policy
- Do not add heuristics to the harness; it exists to surface data that will inform heuristics in Metallic. The harness should remain a neutral executor and reporter.
