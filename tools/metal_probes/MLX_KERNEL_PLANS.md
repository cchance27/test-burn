# MLX Kernel Optimization Comprehensive Plan

## Overview

This plan leverages insights from the MLX dot product example to optimize Metal kernels for the Qwen2.5 shapes, addressing the key bottlenecks identified in the current implementation while maintaining GPU performance parity with MPS.

### Current Harness & Manifest State

- The enhanced Swift harness (`run_matmul_probes_enhanced.sh` pipeline) now loads `variants_enhanced.json`, enforces each variant’s `supports.*` flags, and records any filtered specs as explicit gaps.  
- Updating kernel capabilities therefore requires flipping the matching JSON booleans (`smallK`, `bias`, `accumulate`, `supportedNValues`, etc.) after the Metal implementation is proven.  
- GEMV variants deliberately advertise `bias: false` today; until we land epilogue/bias support in Metal we should leave those specs routed to other backends.

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
